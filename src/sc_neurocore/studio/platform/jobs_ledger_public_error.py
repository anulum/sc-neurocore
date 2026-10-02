# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public job error custody migration

"""Upgrade diagnostic custody without reclassifying historical exception text."""

import sqlite3


def migrate_public_error(connection: sqlite3.Connection) -> None:
    """Add a nullable public projection without changing legacy diagnostics.

    Old error text has no authored provenance. Its new projection stays absent,
    and public readers use their fixed fallback. The owning migration transaction
    commits the column and schema version together.

    Parameters
    ----------
    connection : sqlite3.Connection
        Owning schema transaction with named rows and an existing jobs table.

    Raises
    ------
    sqlite3.Error
        Schema inspection or column creation failed; the caller rolls back.
    """
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(jobs)")}
    if "public_error" not in columns:
        connection.execute("ALTER TABLE jobs ADD COLUMN public_error TEXT")
