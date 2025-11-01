from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def test_core_tables_exist(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    tables = set(inspector.get_table_names())
    expected = {
        "users",
        "transactions",
        "categories",
        "savings_accounts",
        "savings_transactions",
        "wa_messages",
        "ai_audit",
        "alembic_version",
    }
    missing = expected.difference(tables)
    assert not missing, f"Missing tables after migration: {missing}"


def test_transactions_schema_details(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    columns = {col["name"]: col for col in inspector.get_columns("transactions")}

    for required in [
        "direction",
        "status",
        "amount",
        "tx_datetime",
        "user_id",
        "category_id",
        "ai_confidence",
    ]:
        assert required in columns, f"transactions missing column {required}"

    # Ensure numeric precision for amount and timezone awareness for datetime
    amount_type = columns["amount"]["type"]
    assert getattr(amount_type, "precision", None) == 18
    assert getattr(amount_type, "scale", None) == 2

    tx_dt_type = columns["tx_datetime"]["type"]
    assert getattr(tx_dt_type, "timezone", False), "tx_datetime should be timezone aware"

    indexes = {idx["name"] for idx in inspector.get_indexes("transactions")}
    for name in [
        "ix_transactions_user_id",
        "ix_transactions_category_id",
        "ix_transactions_tx_datetime",
        "ix_transactions_user_txdt",
    ]:
        assert name in indexes, f"transactions missing index {name}"


def test_savings_relationships(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    fk_accounts = inspector.get_foreign_keys("savings_accounts")
    assert any(fk["referred_table"] == "users" for fk in fk_accounts)

    fk_transactions = inspector.get_foreign_keys("savings_transactions")
    referenced_tables = {fk["referred_table"] for fk in fk_transactions}
    assert {"savings_accounts", "users"}.issubset(referenced_tables)

    with migrated_engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM information_schema.check_constraints WHERE constraint_name = :name"),
            {"name": "ck_savings_transactions_amount_positive"},
        )
        assert result.fetchone() is not None


def test_wa_messages_indexes(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)
    wa_indexes = {idx["name"] for idx in inspector.get_indexes("wa_messages")}
    for name in [
        "ix_wa_messages_user_id",
        "ix_wa_messages_wa_from",
        "ix_wa_messages_created_at",
    ]:
        assert name in wa_indexes
