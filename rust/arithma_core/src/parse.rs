//! Reading an expression from text.
//!
//! `"2 * pi * radius"` becomes an [`ArithmaExpression`] that
//! [`Evaluable::evaluate`](crate::expression::Evaluable::evaluate) can turn into
//! a number given bindings for `radius`.
//!
//! # Why this is here and not where it is used
//!
//! PlayTow's asset format lets a float be authored as a formula — a scatter
//! radius written `"2 * pi * spacing"` rather than as a number someone worked
//! out by hand. Something has to turn that string into arithmetic, and the
//! obvious place to put it is next to the thing that needs it.
//!
//! That would be wrong, and the capability table says so: symbolic mathematics
//! is this crate's, and a second algebra parser living beside a consumer is how
//! two grammars drift until the same string means different things in two
//! places. This crate already builds expressions programmatically and already
//! evaluates them against bindings; the only missing piece was reading one from
//! text, and the missing piece belongs with the rest of it.
//!
//! # The grammar
//!
//! ```text
//! expr    := term (('+' | '-') term)*
//! term    := unary (('*' | '/' | '%') unary)*
//! unary   := ('-' | '+') unary | power
//! power   := primary ('^' unary)?          -- right-associative
//! primary := number | name | name '(' args ')' | '(' expr ')'
//! args    := expr (',' expr)*
//! ```
//!
//! **`unary` sits ABOVE `power`, and that ordering is the whole subtlety.**
//! Because `unary` descends into `power`, a leading minus applies to the
//! finished power: `-2^2` is `-(2^2) = -4`, not `(-2)^2 = 4`. Because `power`
//! takes a `unary` as its *exponent*, `2^-1` is `0.5` rather than a parse
//! error. And because that exponent recurses back through `unary` into `power`,
//! `2^3^2` is `2^(3^2) = 512` rather than `(2^3)^2 = 64`.
//!
//! Writing it the other way round -- `power := unary ('^' power)?`, which reads
//! perfectly naturally -- makes `-2^2` evaluate to 4. That is the classic
//! precedence-climbing bug and it is silent: every wrong answer it produces is
//! a plausible number. It was written that way here first, and the test below
//! is what caught it.

use crate::expression::ArithmaExpression;
use crate::function::ArithmaFunction;

/// The text could not be read as an expression.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// The input had nothing in it.
    #[error("there is no expression here")]
    Empty,

    /// A character that cannot appear in an expression.
    #[error("`{found}` at position {at} is not something an expression can contain")]
    BadCharacter {
        /// The offending character.
        found: char,
        /// Where it was, as a byte offset.
        at: usize,
    },

    /// Something was expected and something else was there.
    #[error("expected {expected} at position {at}, found {found}")]
    Expected {
        /// What the grammar wanted.
        expected: &'static str,
        /// What was actually there.
        found: String,
        /// Where, as a byte offset.
        at: usize,
    },

    /// Input ended in the middle of something.
    #[error("the expression ends after `{after}`, and it is not finished")]
    UnexpectedEnd {
        /// The last thing successfully read.
        after: String,
    },

    /// Text after a complete expression.
    #[error("`{rest}` is left over after the expression ends at position {at}")]
    TrailingInput {
        /// What was left.
        rest: String,
        /// Where the expression ended.
        at: usize,
    },

    /// A function name nothing is registered for.
    #[error("`{name}` is not a function this understands; known: {known}")]
    UnknownFunction {
        /// What was written.
        name: String,
        /// What could have been written.
        known: String,
    },

    /// A function called with the wrong number of arguments.
    #[error("`{name}` takes {wanted} argument(s) and was given {given}")]
    WrongArity {
        /// The function.
        name: String,
        /// How many it takes.
        wanted: usize,
        /// How many it got.
        given: usize,
    },
}

/// One lexical item.
#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Self::Number(n) => format!("the number {n}"),
            Self::Name(n) => format!("`{n}`"),
            Self::Plus => "`+`".into(),
            Self::Minus => "`-`".into(),
            Self::Star => "`*`".into(),
            Self::Slash => "`/`".into(),
            Self::Percent => "`%`".into(),
            Self::Caret => "`^`".into(),
            Self::LParen => "`(`".into(),
            Self::RParen => "`)`".into(),
            Self::Comma => "`,`".into(),
        }
    }
}

/// Split the text into tokens.
fn lex(text: &str) -> Result<Vec<(Token, usize)>, ParseError> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        let start = i;
        let token = match c {
            '+' => {
                i += 1;
                Token::Plus
            }
            '-' => {
                i += 1;
                Token::Minus
            }
            '*' => {
                i += 1;
                Token::Star
            }
            '/' => {
                i += 1;
                Token::Slash
            }
            '%' => {
                i += 1;
                Token::Percent
            }
            '^' => {
                i += 1;
                Token::Caret
            }
            '(' => {
                i += 1;
                Token::LParen
            }
            ')' => {
                i += 1;
                Token::RParen
            }
            ',' => {
                i += 1;
                Token::Comma
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut j = i;
                while j < bytes.len() && ((bytes[j] as char).is_ascii_digit() || bytes[j] == b'.') {
                    j += 1;
                }
                // An exponent, but only when it really is one: `2e5` is a
                // number and `2 * e` is a product with Euler's constant, and
                // the difference is whether a digit or sign follows the `e`.
                if j < bytes.len() && (bytes[j] == b'e' || bytes[j] == b'E') {
                    let mut k = j + 1;
                    if k < bytes.len() && (bytes[k] == b'+' || bytes[k] == b'-') {
                        k += 1;
                    }
                    if k < bytes.len() && (bytes[k] as char).is_ascii_digit() {
                        while k < bytes.len() && (bytes[k] as char).is_ascii_digit() {
                            k += 1;
                        }
                        j = k;
                    }
                }
                let literal = &text[i..j];
                let value = literal.parse::<f64>().map_err(|_| ParseError::Expected {
                    expected: "a number",
                    found: format!("`{literal}`"),
                    at: i,
                })?;
                i = j;
                Token::Number(value)
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut j = i;
                while j < bytes.len() {
                    let ch = bytes[j] as char;
                    if ch.is_alphanumeric() || ch == '_' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                let name = text[i..j].to_string();
                i = j;
                Token::Name(name)
            }
            other => {
                return Err(ParseError::BadCharacter {
                    found: other,
                    at: i,
                })
            }
        };
        out.push((token, start));
    }

    if out.is_empty() {
        return Err(ParseError::Empty);
    }
    Ok(out)
}

struct Parser<'a> {
    tokens: &'a [(Token, usize)],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn at(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map(|(_, a)| *a)
            .unwrap_or_else(|| self.tokens.last().map(|(_, a)| *a).unwrap_or(0))
    }

    fn last_described(&self) -> String {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|(t, _)| t.describe())
            .unwrap_or_else(|| "the start".into())
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).map(|(t, _)| t.clone());
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, want: &Token, expected: &'static str) -> Result<(), ParseError> {
        match self.peek() {
            Some(t) if t == want => {
                self.pos += 1;
                Ok(())
            }
            Some(t) => Err(ParseError::Expected {
                expected,
                found: t.describe(),
                at: self.at(),
            }),
            None => Err(ParseError::UnexpectedEnd {
                after: self.last_described(),
            }),
        }
    }

    fn expr(&mut self) -> Result<ArithmaExpression, ParseError> {
        let mut left = self.term()?;
        while let Some(op) = self.peek().cloned() {
            match op {
                Token::Plus => {
                    self.pos += 1;
                    left = ArithmaExpression::add(left, self.term()?);
                }
                Token::Minus => {
                    self.pos += 1;
                    left = ArithmaExpression::sub(left, self.term()?);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<ArithmaExpression, ParseError> {
        let mut left = self.unary()?;
        while let Some(op) = self.peek().cloned() {
            match op {
                Token::Star => {
                    self.pos += 1;
                    left = ArithmaExpression::mul(left, self.unary()?);
                }
                Token::Slash => {
                    self.pos += 1;
                    left = ArithmaExpression::div(left, self.unary()?);
                }
                Token::Percent => {
                    self.pos += 1;
                    let right = self.unary()?;
                    // a % b == a - b * floor(a / b), built from what the
                    // expression tree already has rather than adding a Mod
                    // function nothing else would use.
                    let q = ArithmaExpression::div(left.clone(), right.clone());
                    let f = ArithmaExpression::func(ArithmaFunction::Floor, vec![q]);
                    left = ArithmaExpression::sub(left, ArithmaExpression::mul(right, f));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// A sign, applied to a finished power.
    ///
    /// Descending into [`Parser::power`] is what makes `-2^2` equal `-4`: the
    /// power is built first and the negation wraps the result. The natural-
    /// looking alternative, having `power` call `unary` for its base, silently
    /// gives 4.
    fn unary(&mut self) -> Result<ArithmaExpression, ParseError> {
        match self.peek() {
            Some(Token::Minus) => {
                self.pos += 1;
                Ok(ArithmaExpression::neg(self.unary()?))
            }
            Some(Token::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    /// `^`, right-associative, with a `unary` exponent.
    ///
    /// The exponent recurses through [`Parser::unary`] rather than straight
    /// back into `power`, which buys two things at once: `2^-1` parses, and
    /// `2^3^2` is `2^(3^2) = 512` rather than `(2^3)^2 = 64`.
    fn power(&mut self) -> Result<ArithmaExpression, ParseError> {
        let base = self.primary()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.pos += 1;
            let exponent = self.unary()?;
            return Ok(ArithmaExpression::pow(base, exponent));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<ArithmaExpression, ParseError> {
        let at = self.at();
        match self.bump() {
            Some(Token::Number(n)) => Ok(ArithmaExpression::from_f64(n)),
            Some(Token::LParen) => {
                let inner = self.expr()?;
                self.expect(&Token::RParen, "a closing `)`")?;
                Ok(inner)
            }
            Some(Token::Name(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.pos += 1;
                    let mut args = vec![self.expr()?];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.pos += 1;
                        args.push(self.expr()?);
                    }
                    self.expect(&Token::RParen, "a closing `)`")?;
                    return build_call(&name, args);
                }
                // A bare name is a variable. Constants like `pi` and `e` are
                // NOT special-cased here: the caller binds them, so a project
                // can define its own and there is one place a name is resolved
                // rather than two that can disagree.
                Ok(ArithmaExpression::var(&name))
            }
            Some(other) => Err(ParseError::Expected {
                expected: "a number, a name, or `(`",
                found: other.describe(),
                at,
            }),
            None => Err(ParseError::UnexpectedEnd {
                after: self.last_described(),
            }),
        }
    }
}

/// Every function a written expression may call.
const FUNCTIONS: &[(&str, ArithmaFunction, usize)] = &[
    ("abs", ArithmaFunction::Abs, 1),
    ("acos", ArithmaFunction::Acos, 1),
    ("asin", ArithmaFunction::Asin, 1),
    ("atan", ArithmaFunction::Atan, 1),
    ("atan2", ArithmaFunction::Atan2, 2),
    ("cbrt", ArithmaFunction::Cbrt, 1),
    ("ceil", ArithmaFunction::Ceil, 1),
    ("cos", ArithmaFunction::Cos, 1),
    ("cosh", ArithmaFunction::Cosh, 1),
    ("exp", ArithmaFunction::Exp, 1),
    ("floor", ArithmaFunction::Floor, 1),
    ("ln", ArithmaFunction::Ln, 1),
    ("log", ArithmaFunction::Log, 1),
    ("log10", ArithmaFunction::Log10, 1),
    ("log2", ArithmaFunction::Log2, 1),
    ("round", ArithmaFunction::Round, 1),
    ("sign", ArithmaFunction::Sign, 1),
    ("sin", ArithmaFunction::Sin, 1),
    ("sinh", ArithmaFunction::Sinh, 1),
    ("sqrt", ArithmaFunction::Sqrt, 1),
    ("tan", ArithmaFunction::Tan, 1),
    ("tanh", ArithmaFunction::Tanh, 1),
];

fn build_call(name: &str, args: Vec<ArithmaExpression>) -> Result<ArithmaExpression, ParseError> {
    let lower = name.to_ascii_lowercase();
    let Some((_, function, arity)) = FUNCTIONS.iter().find(|(n, _, _)| *n == lower) else {
        return Err(ParseError::UnknownFunction {
            name: name.to_string(),
            known: FUNCTIONS
                .iter()
                .map(|(n, _, _)| *n)
                .collect::<Vec<_>>()
                .join(", "),
        });
    };
    if args.len() != *arity {
        return Err(ParseError::WrongArity {
            name: lower,
            wanted: *arity,
            given: args.len(),
        });
    }
    Ok(ArithmaExpression::func(function.clone(), args))
}

/// Read an expression from text.
///
/// Names are left as variables — including `pi` and `e`. Binding them is the
/// caller's job, so a project can define its own constants and a name is
/// resolved in exactly one place rather than in two that can disagree.
pub fn parse_expression(text: &str) -> Result<ArithmaExpression, ParseError> {
    let tokens = lex(text)?;
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
    };
    let expr = parser.expr()?;
    if parser.pos < tokens.len() {
        let (_, at) = tokens[parser.pos];
        return Err(ParseError::TrailingInput {
            rest: text[at..].trim().to_string(),
            at,
        });
    }
    Ok(expr)
}

/// Every name a parsed expression refers to, sorted and deduplicated.
///
/// What a caller needs in order to know which bindings to supply — and what
/// lets "you did not bind `radius`" be said before evaluation rather than
/// during it.
pub fn free_variables(expr: &ArithmaExpression) -> Vec<String> {
    let mut out = Vec::new();
    collect_variables(expr, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_variables(expr: &ArithmaExpression, out: &mut Vec<String>) {
    match expr {
        ArithmaExpression::Variable(name) => out.push(name.clone()),
        ArithmaExpression::Function(_, args) => {
            for arg in args {
                collect_variables(arg, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ArithmaBindings, Evaluable};

    fn eval(text: &str) -> f64 {
        let expr = parse_expression(text).unwrap_or_else(|e| panic!("`{text}`: {e}"));
        expr.evaluate(&ArithmaBindings::new())
            .unwrap_or_else(|e| panic!("`{text}`: {e}"))
    }

    fn eval_with(text: &str, pairs: &[(&str, f64)]) -> f64 {
        let mut b = ArithmaBindings::new();
        for (k, v) in pairs {
            b.insert((*k).to_string(), *v);
        }
        parse_expression(text).unwrap().evaluate(&b).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert!(close(eval("1 + 2"), 3.0));
        assert!(close(eval("2 + 3 * 4"), 14.0), "times binds tighter");
        assert!(close(eval("(2 + 3) * 4"), 20.0));
        assert!(close(eval("10 / 4"), 2.5));
        assert!(close(eval("10 - 3 - 2"), 5.0), "minus is left-associative");
        assert!(close(eval("100 / 10 / 2"), 5.0), "so is divide");
    }

    #[test]
    fn power_is_right_associative_and_beats_unary_minus_on_its_left() {
        // The classic precedence-climbing bug, and it is silent: every wrong
        // answer here is a plausible number.
        assert!(close(eval("2 ^ 3 ^ 2"), 512.0), "2^(3^2), not (2^3)^2");
        assert!(close(eval("-2 ^ 2"), -4.0), "-(2^2), not (-2)^2");
        assert!(
            close(eval("2 ^ -1"), 0.5),
            "a negative exponent still parses"
        );
    }

    #[test]
    fn unary_minus_stacks_and_unary_plus_is_a_no_op() {
        assert!(close(eval("--3"), 3.0));
        assert!(close(eval("-(-3)"), 3.0));
        assert!(close(eval("+4"), 4.0));
        assert!(close(eval("3 - -2"), 5.0));
    }

    #[test]
    fn numbers_in_every_spelling_a_person_writes() {
        assert!(close(eval("1.5"), 1.5));
        assert!(close(eval(".5"), 0.5));
        assert!(close(eval("2e3"), 2000.0));
        assert!(close(eval("2e-3"), 0.002));
        assert!(close(eval("1E2"), 100.0));
    }

    #[test]
    fn e_after_a_number_is_an_exponent_only_when_a_digit_follows() {
        // `2e5` is a number; `2 * e` is a product with a variable called `e`.
        // Getting this wrong makes one of the two a parse error, and which one
        // is not something a user could guess.
        assert!(close(eval("2e5"), 200000.0));
        // `e` is bound as an ordinary variable here -- the subject of the test
        // is the lexer, not the constant -- so it takes a value that cannot be
        // mistaken for an approximation of Euler's number.
        assert!(close(eval_with("2 * e", &[("e", 3.0)]), 6.0));
        assert!(close(eval_with("e", &[("e", 1.0)]), 1.0));
    }

    #[test]
    fn variables_come_from_the_bindings() {
        assert!(close(eval_with("2 * radius", &[("radius", 21.0)]), 42.0));
        assert!(close(
            eval_with("2 * pi * r", &[("pi", std::f64::consts::PI), ("r", 1.0)]),
            std::f64::consts::TAU
        ));
    }

    #[test]
    fn pi_is_not_special_cased_so_a_project_can_define_its_own_constants() {
        // One place a name is resolved, rather than two that can disagree. A
        // built-in `pi` plus a bound `pi` is exactly the kind of shadowing that
        // produces a number nobody can account for.
        let expr = parse_expression("pi").unwrap();
        assert_eq!(free_variables(&expr), vec!["pi"]);
        assert!(
            expr.evaluate(&ArithmaBindings::new()).is_err(),
            "unbound, rather than silently 3.14159"
        );
    }

    #[test]
    fn functions_of_one_and_two_arguments() {
        assert!(close(eval("sqrt(16)"), 4.0));
        assert!(close(eval("abs(-3)"), 3.0));
        assert!(close(eval("floor(2.7)"), 2.0));
        assert!(close(eval("ceil(2.1)"), 3.0));
        assert!(close(eval("sin(0)"), 0.0));
        assert!(close(eval("atan2(0, 1)"), 0.0));
        assert!(close(eval("SQRT(9)"), 3.0), "the name is case-insensitive");
    }

    #[test]
    fn functions_nest_and_take_expressions() {
        assert!(close(eval("sqrt(2 * 8)"), 4.0));
        assert!(close(eval("abs(floor(-2.5))"), 3.0));
        assert!(close(eval_with("sqrt(r * r)", &[("r", 5.0)]), 5.0));
    }

    #[test]
    fn modulo_is_built_from_what_the_tree_already_has() {
        assert!(close(eval("7 % 3"), 1.0));
        assert!(close(eval("7.5 % 2"), 1.5));
    }

    #[test]
    fn an_unknown_function_lists_the_ones_that_exist() {
        let err = parse_expression("frobnicate(2)").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("frobnicate"), "{msg}");
        assert!(msg.contains("sqrt") && msg.contains("sin"), "{msg}");
    }

    #[test]
    fn the_wrong_number_of_arguments_says_how_many_it_wanted() {
        let err = parse_expression("sqrt(1, 2)").unwrap_err();
        assert!(
            matches!(
                err,
                ParseError::WrongArity {
                    wanted: 1,
                    given: 2,
                    ..
                }
            ),
            "{err}"
        );
        assert!(parse_expression("atan2(1)").is_err());
    }

    #[test]
    fn malformed_input_is_an_error_that_says_where() {
        assert!(matches!(parse_expression(""), Err(ParseError::Empty)));
        assert!(matches!(parse_expression("   "), Err(ParseError::Empty)));

        let err = parse_expression("2 +").unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEnd { .. }), "{err}");

        let err = parse_expression("(2 + 3").unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEnd { .. }), "{err}");

        let err = parse_expression("2 $ 3").unwrap_err();
        assert!(
            matches!(err, ParseError::BadCharacter { found: '$', .. }),
            "{err}"
        );

        let err = parse_expression("2 3").unwrap_err();
        assert!(matches!(err, ParseError::TrailingInput { .. }), "{err}");
    }

    #[test]
    fn free_variables_are_listed_once_and_in_order() {
        // What lets "you did not bind `radius`" be said before evaluation
        // rather than during it.
        let expr = parse_expression("a * b + sqrt(a) - c").unwrap();
        assert_eq!(free_variables(&expr), vec!["a", "b", "c"]);
        assert!(free_variables(&parse_expression("1 + 2").unwrap()).is_empty());
    }

    #[test]
    fn whitespace_is_irrelevant() {
        assert!(close(eval("2*3+4"), eval("  2 * 3   +  4 ")));
    }

    #[test]
    fn the_function_table_is_sorted_and_every_entry_parses() {
        // Sorted so the "known:" list in an error reads sensibly, and every
        // entry exercised so a name in the table that the evaluator does not
        // implement is caught here rather than by a user.
        let names: Vec<&str> = FUNCTIONS.iter().map(|(n, _, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "the table is out of order");

        for (name, _, arity) in FUNCTIONS {
            let args = vec!["1"; *arity].join(", ");
            let text = format!("{name}({args})");
            parse_expression(&text).unwrap_or_else(|e| panic!("`{text}`: {e}"));
        }
    }
}
