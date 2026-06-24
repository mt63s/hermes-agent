# Honcho Slice 2 Reconciler Contract

Slice 2 MUST read `$HERMES_HOME/honcho_untagged_ingest_exclusions.jsonl` as a deduplicated set of `hermes_message_id` values; repeated or duplicate JSONL records are idempotent and must not affect gap detection beyond excluding that message id once.
