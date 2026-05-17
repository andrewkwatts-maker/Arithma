//====== Arithma/rust/arithma_core/src/external/mod.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! External-function registry â€” the integration seam for non-Arithmos backends.
//!
//! The registry is the foundation of the three-way routing described in plan
//! Â§B.5: pt-arithmos engine glue, Arithmos (default), and EML-Math co-exist by
//! registering themselves as backends here. Downstream plugins (pt-eml-bridge)
//! call into the registry rather than dispatching directly.

pub mod registry;
pub mod sdk;
pub mod config;

#[cfg(feature = "cpp-support")]
pub mod cpp_executor;

#[cfg(feature = "rust-support")]
pub mod rust_executor;

