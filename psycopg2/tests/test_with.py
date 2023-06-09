#!/usr/bin/env python

# test_ctxman.py - unit test for connection and cursor used as context manager
#
# Copyright (C) 2012-2019 Daniele Varrazzo  <daniele.varrazzo@gmail.com>
# Copyright (C) 2020-2021 The Psycopg Team
#
# psycopg2 is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# In addition, as a special exception, the copyright holders give
# permission to link this program with the OpenSSL library (or with
# modified versions of OpenSSL that use the same license as OpenSSL),
# and distribute linked combinations including the two.
#
# You must obey the GNU Lesser General Public License in all respects for
# all of the code used other than OpenSSL.
#
# psycopg2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.

import psycopg2
import psycopg2.extensions as ext

import unittest
from .testutils import ConnectingTestCase, skip_before_postgres, skip_if_crdb


class WithTestCase(ConnectingTestCase):
    def setUp(self):
        ConnectingTestCase.setUp(self)
        curs = self.conn.cursor()
        try:
            curs.execute("delete from test_with")
            self.conn.commit()
        except psycopg2.ProgrammingError:
            # assume table doesn't exist
            self.conn.rollback()
            curs.execute("create table test_with (id integer primary key)")
            self.conn.commit()


class WithConnectionTestCase(WithTestCase):
    def test_with_ok(self):
        with self.conn as conn:
            self.assert_(self.conn is conn)
            self.assertEqual(conn.status, ext.STATUS_READY)
            curs = conn.cursor()
            curs.execute("insert into test_with values (1)")
            self.assertEqual(conn.status, ext.STATUS_BEGIN)

        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [(1,)])

    def test_with_connect_idiom(self):
        with self.connect() as conn:
            self.assertEqual(conn.status, ext.STATUS_READY)
            curs = conn.cursor()
            curs.execute("insert into test_with values (2)")
            self.assertEqual(conn.status, ext.STATUS_BEGIN)

        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [(2,)])

    def test_with_error_db(self):
        def f():
            with self.conn as conn:
                curs = conn.cursor()
                curs.execute("insert into test_with values ('a')")

        self.assertRaises(psycopg2.DataError, f)
        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [])

    def test_with_error_python(self):
        def f():
            with self.conn as conn:
                curs = conn.cursor()
                curs.execute("insert into test_with values (3)")
                1 / 0

        self.assertRaises(ZeroDivisionError, f)
        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [])

    def test_with_closed(self):
        def f():
            with self.conn:
                pass

        self.conn.close()
        self.assertRaises(psycopg2.InterfaceError, f)

    def test_subclass_commit(self):
        commits = []

        class MyConn(ext.connection):
            def commit(self):
                commits.append(None)
                super().commit()

        with self.connect(connection_factory=MyConn) as conn:
            curs = conn.cursor()
            curs.execute("insert into test_with values (10)")

        self.assertEqual(conn.status, ext.STATUS_READY)
        self.assert_(commits)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [(10,)])

    def test_subclass_rollback(self):
        rollbacks = []

        class MyConn(ext.connection):
            def rollback(self):
                rollbacks.append(None)
                super().rollback()

        try:
            with self.connect(connection_factory=MyConn) as conn:
                curs = conn.cursor()
                curs.execute("insert into test_with values (11)")
                1 / 0
        except ZeroDivisionError:
            pass
        else:
            self.assert_("exception not raised")

        self.assertEqual(conn.status, ext.STATUS_READY)
        self.assert_(rollbacks)

        curs = conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [])

    def test_cant_reenter(self):
        raised_ok = False
        with self.conn:
            try:
                with self.conn:
                    pass
            except psycopg2.ProgrammingError:
                raised_ok = True

        self.assert_(raised_ok)

        # Still good
        with self.conn:
            pass

    def test_with_autocommit(self):
        self.conn.autocommit = True
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )
        with self.conn:
            curs = self.conn.cursor()
            curs.execute("insert into test_with values (1)")
            self.assertEqual(
                self.conn.info.transaction_status,
                ext.TRANSACTION_STATUS_INTRANS,
            )

        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )
        curs.execute("select count(*) from test_with")
        self.assertEqual(curs.fetchone()[0], 1)
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )

    def test_with_autocommit_pyerror(self):
        self.conn.autocommit = True
        raised_ok = False
        try:
            with self.conn:
                curs = self.conn.cursor()
                curs.execute("insert into test_with values (2)")
                self.assertEqual(
                    self.conn.info.transaction_status,
                    ext.TRANSACTION_STATUS_INTRANS,
                )
                1 / 0
        except ZeroDivisionError:
            raised_ok = True

        self.assert_(raised_ok)
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )
        curs.execute("select count(*) from test_with")
        self.assertEqual(curs.fetchone()[0], 0)
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )

    def test_with_autocommit_pgerror(self):
        self.conn.autocommit = True
        raised_ok = False
        try:
            with self.conn:
                curs = self.conn.cursor()
                curs.execute("insert into test_with values (2)")
                self.assertEqual(
                    self.conn.info.transaction_status,
                    ext.TRANSACTION_STATUS_INTRANS,
                )
                curs.execute("insert into test_with values ('x')")
        except psycopg2.errors.InvalidTextRepresentation:
            raised_ok = True

        self.assert_(raised_ok)
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )
        curs.execute("select count(*) from test_with")
        self.assertEqual(curs.fetchone()[0], 0)
        self.assertEqual(
            self.conn.info.transaction_status, ext.TRANSACTION_STATUS_IDLE
        )


class WithCursorTestCase(WithTestCase):
    def test_with_ok(self):
        with self.conn as conn:
            with conn.cursor() as curs:
                curs.execute("insert into test_with values (4)")
                self.assert_(not curs.closed)
            self.assertEqual(self.conn.status, ext.STATUS_BEGIN)
            self.assert_(curs.closed)

        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [(4,)])

    def test_with_error(self):
        try:
            with self.conn as conn:
                with conn.cursor() as curs:
                    curs.execute("insert into test_with values (5)")
                    1 / 0
        except ZeroDivisionError:
            pass

        self.assertEqual(self.conn.status, ext.STATUS_READY)
        self.assert_(not self.conn.closed)
        self.assert_(curs.closed)

        curs = self.conn.cursor()
        curs.execute("select * from test_with")
        self.assertEqual(curs.fetchall(), [])

    def test_subclass(self):
        closes = []

        class MyCurs(ext.cursor):
            def close(self):
                closes.append(None)
                super().close()

        with self.conn.cursor(cursor_factory=MyCurs) as curs:
            self.assert_(isinstance(curs, MyCurs))

        self.assert_(curs.closed)
        self.assert_(closes)

    @skip_if_crdb("named cursor", version="< 22.1")
    def test_exception_swallow(self):
        # bug #262: __exit__ calls cur.close() that hides the exception
        # with another error.
        try:
            with self.conn as conn:
                with conn.cursor('named') as cur:
                    cur.execute("select 1/0")
                    cur.fetchone()
        except psycopg2.DataError as e:
            self.assertEqual(e.pgcode, '22012')
        else:
            self.fail("where is my exception?")

    @skip_if_crdb("named cursor", version="< 22.1")
    @skip_before_postgres(8, 2)
    def test_named_with_noop(self):
        with self.conn.cursor('named'):
            pass


def test_suite():
    return unittest.TestLoader().loadTestsFromName(__name__)


if __name__ == "__main__":
    unittest.main()
