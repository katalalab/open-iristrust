//! Deterministic fingerprints prevent mixing different encoding configurations.
use crate::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
pub(crate) fn fingerprint(domain: &str, config: &impl Serialize) -> Result<String> {
    let hash = Sha256::digest(serde_json::to_vec(config)?);
    Ok(format!("{domain}:{hash:x}"))
}
