# Document 07: metrics

## Purpose
Counters and timers.

## Contract
Callers pass records as plain dictionaries. The module never mutates the caller's
object; it copies before writing. Reads are not cached at this layer -- the cache
module owns that, and caching here as well was the cause of two stale-read bugs.

## Known limits
- No pagination beyond 500 records in a single call
- Timestamps are stored as floats and compared without rounding
- Failures raise, they do not return sentinel values

## Change history
- Added strict validation on put()
- Removed the implicit default owner
- Split the export helpers out of this module
