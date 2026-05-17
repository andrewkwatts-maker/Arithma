//====== Arithma/rust/arithma_core/src/lookup/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! Lookup tables for expensive transcendental functions.
//!
//! Two tables are exposed:
//! - [`trig_hash`] â€” sin/cos/tan with hash-keyed canonical-angle mapping.
//! - [`math_hash`] â€” exp/ln/sqrt/etc. by hashed argument bucket.
//!
//! Wave-2 stub. The real tables migrate from pt-arithmos in Wave 3.

pub mod trig_hash;
pub mod math_hash;

