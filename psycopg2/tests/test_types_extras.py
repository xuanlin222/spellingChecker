#!/usr/bin/env python
#
# types_extras.py - tests for extras types conversions
#
# Copyright (C) 2008-2019 Federico Di Gregorio  <fog@debian.org>
# Copyright (C) 2020-2021 The Psycopg Team
#
# psycopg2 is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# psycopg2 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.

import re
import json
import uuid
import warnings
from decimal import Decimal
from datetime import date, datetime, timedelta, timezone
from functools import wraps
from pickle import dumps, loads

import unittest
from .testutils import (skip_if_no_uuid, skip_before_postgres,
    ConnectingTestCase, raises_typeerror, slow,
    restore_types, skip_if_crdb, crdb_version)

import psycopg2
import psycopg2.extras
import psycopg2.extensions as ext
from psycopg2._json import _get_json_oids
from psycopg2.extras import (
    CompositeCaster, DateRange, DateTimeRange, DateTimeTZRange, HstoreAdapter,
    Inet, Json, NumericRange, Range, RealDictConnection,
    register_composite, register_hstore, register_range,
)


class TypesExtrasTests(ConnectingTestCase):
    """Test that all type conversions are working."""

    def execute(self, *args):
        curs = self.conn.cursor()
        curs.execute(*args)
        return curs.fetchone()[0]

    @skip_if_no_uuid
    def testUUID(self):
        psycopg2.extras.register_uuid()
        u = uuid.UUID('9c6d5a77-7256-457e-9461-347b4358e350')
        s = self.execute("SELECT %s AS foo", (u,))
        self.failUnless(u == s)
        # must survive NULL cast to a uuid
        s = self.execute("SELECT NULL::uuid AS foo")
        self.failUnless(s is None)

    @skip_if_no_uuid
    def testUUIDARRAY(self):
        psycopg2.extras.register_uuid()
        u = [uuid.UUID('9c6d5a77-7256-457e-9461-347b4358e350'),
             uuid.UUID('9c6d5a77-7256-457e-9461-347b4358e352')]
        s = self.execute("SELECT %s AS foo", (u,))
        self.failUnless(u == s)
        # array with a NULL element
        u = [uuid.UUID('9c6d5a77-7256-457e-9461-347b4358e350'), None]
        s = self.execute("SELECT %s AS foo", (u,))
        self.failUnless(u == s)
        # must survive NULL cast to a uuid[]
        s = self.execute("SELECT NULL::uuid[] AS foo")
        self.failUnless(s is None)
        # what about empty arrays?
        s = self.execute("SELECT '{}'::uuid[] AS foo")
        self.failUnless(type(s) == list and len(s) == 0)

    @restore_types
    def testINET(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            psycopg2.extras.register_inet()

        i = psycopg2.extras.Inet("192.168.1.0/24")
        s = self.execute("SELECT %s AS foo", (i,))
        self.failUnless(i.addr == s.addr)
        # must survive NULL cast to inet
        s = self.execute("SELECT NULL::inet AS foo")
        self.failUnless(s is None)

    @restore_types
    def testINETARRAY(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            psycopg2.extras.register_inet()

        i = psycopg2.extras.Inet("192.168.1.0/24")
        s = self.execute("SELECT %s AS foo", ([i],))
        self.failUnless(i.addr == s[0].addr)
        # must survive NULL cast to inet
        s = self.execute("SELECT NULL::inet[] AS foo")
        self.failUnless(s is None)

    def test_inet_conform(self):
        i = Inet("192.168.1.0/24")
        a = psycopg2.extensions.adapt(i)
        a.prepare(self.conn)
        self.assertQuotedEqual(a.getquoted(), b"'192.168.1.0/24'::inet")

        # adapts ok with unicode too
        i = Inet("192.168.1.0/24")
        a = psycopg2.extensions.adapt(i)
        a.prepare(self.conn)
        self.assertQuotedEqual(a.getquoted(), b"'192.168.1.0/24'::inet")

    def test_adapt_fail(self):
        class Foo:
            pass
        self.assertRaises(psycopg2.ProgrammingError,
            psycopg2.extensions.adapt, Foo(), ext.ISQLQuote, None)
        try:
            psycopg2.extensions.adapt(Foo(), ext.ISQLQuote, None)
        except psycopg2.ProgrammingError as err:
            self.failUnless(str(err) == "can't adapt type 'Foo'")

    def test_point_array(self):
        # make sure a point array is never casted to a float array,
        # see https://github.com/psycopg/psycopg2/issues/613
        s = self.execute("""SELECT '{"(1,2)","(3,4)"}' AS foo""")
        self.failUnless(s == """{"(1,2)","(3,4)"}""")


def skip_if_no_hstore(f):
    @wraps(f)
    @skip_if_crdb("hstore")
    def skip_if_no_hstore_(self):
        oids = HstoreAdapter.get_oids(self.conn)
        if oids is None or not oids[0]:
            return self.skipTest("hstore not available in test database")
        return f(self)

    return skip_if_no_hstore_


class HstoreTestCase(ConnectingTestCase):
    def test_adapt_8(self):
        if self.conn.info.server_version >= 90000:
            return self.skipTest("skipping dict adaptation with PG pre-9 syntax")

        o = {'a': '1', 'b': "'", 'c': None}
        if self.conn.encoding == 'UTF8':
            o['d'] = '\xe0'

        a = HstoreAdapter(o)
        a.prepare(self.conn)
        q = a.getquoted()

        self.assert_(q.startswith(b"(("), q)
        ii = q[1:-1].split(b"||")
        ii.sort()

        self.assertEqual(len(ii), len(o))
        self.assertQuotedEqual(ii[0], b"('a' => '1')")
        self.assertQuotedEqual(ii[1], b"('b' => '''')")
        self.assertQuotedEqual(ii[2], b"('c' => NULL)")
        if 'd' in o:
            encc = '\xe0'.encode(psycopg2.extensions.encodings[self.conn.encoding])
            self.assertQuotedEqual(ii[3], b"('d' => '" + encc + b"')")

    def test_adapt_9(self):
        if self.conn.info.server_version < 90000:
            return self.skipTest("skipping dict adaptation with PG 9 syntax")

        o = {'a': '1', 'b': "'", 'c': None}
        if self.conn.encoding == 'UTF8':
            o['d'] = '\xe0'

        a = HstoreAdapter(o)
        a.prepare(self.conn)
        q = a.getquoted()

        m = re.match(br'hstore\(ARRAY\[([^\]]+)\], ARRAY\[([^\]]+)\]\)', q)
        self.assert_(m, repr(q))

        kk = m.group(1).split(b",")
        vv = m.group(2).split(b",")
        ii = list(zip(kk, vv))
        ii.sort()

        self.assertEqual(len(ii), len(o))
        self.assertQuotedEqual(ii[0][0], b"'a'")
        self.assertQuotedEqual(ii[0][1], b"'1'")
        self.assertQuotedEqual(ii[1][0], b"'b'")
        self.assertQuotedEqual(ii[1][1], b"''''")
        self.assertQuotedEqual(ii[2][0], b"'c'")
        self.assertQuotedEqual(ii[2][1], b"NULL")
        if 'd' in o:
            encc = '\xe0'.encode(psycopg2.extensions.encodings[self.conn.encoding])
            self.assertQuotedEqual(ii[3][0], b"'d'")
            self.assertQuotedEqual(ii[3][1], b"'" + encc + b"'")

    def test_parse(self):
        def ok(s, d):
            self.assertEqual(HstoreAdapter.parse(s, None), d)

        ok(None, None)
        ok('', {})
        ok('"a"=>"1", "b"=>"2"', {'a': '1', 'b': '2'})
        ok('"a"  => "1" , "b"  =>  "2"', {'a': '1', 'b': '2'})
        ok('"a"=>NULL, "b"=>"2"', {'a': None, 'b': '2'})
        ok(r'"a"=>"\"", "\""=>"2"', {'a': '"', '"': '2'})
        ok('"a"=>"\'", "\'"=>"2"', {'a': "'", "'": '2'})
        ok('"a"=>"1", "b"=>NULL', {'a': '1', 'b': None})
        ok(r'"a\\"=>"1"', {'a\\': '1'})
        ok(r'"a\""=>"1"', {'a"': '1'})
        ok(r'"a\\\""=>"1"', {r'a\"': '1'})
        ok(r'"a\\\\\""=>"1"', {r'a\\"': '1'})

        def ko(s):
            self.assertRaises(psycopg2.InterfaceError,
                HstoreAdapter.parse, s, None)

        ko('a')
        ko('"a"')
        ko(r'"a\\""=>"1"')
        ko(r'"a\\\\""=>"1"')
        ko('"a=>"1"')
        ko('"a"=>"1", "b"=>NUL')

    @skip_if_no_hstore
    def test_register_conn(self):
        register_hstore(self.conn)
        cur = self.conn.cursor()
        cur.execute("select null::hstore, ''::hstore, 'a => b'::hstore")
        t = cur.fetchone()
        self.assert_(t[0] is None)
        self.assertEqual(t[1], {})
        self.assertEqual(t[2], {'a': 'b'})

    @skip_if_no_hstore
    def test_register_curs(self):
        cur = self.conn.cursor()
        register_hstore(cur)
        cur.execute("select null::hstore, ''::hstore, 'a => b'::hstore")
        t = cur.fetchone()
        self.assert_(t[0] is None)
        self.assertEqual(t[1], {})
        self.assertEqual(t[2], {'a': 'b'})

    @skip_if_no_hstore
    @restore_types
    def test_register_globally(self):
        HstoreAdapter.get_oids(self.conn)
        register_hstore(self.conn, globally=True)
        conn2 = self.connect()
        try:
            cur2 = self.conn.cursor()
            cur2.execute("select 'a => b'::hstore")
            r = cur2.fetchone()
            self.assert_(isinstance(r[0], dict))
        finally:
            conn2.close()

    @skip_if_no_hstore
    def test_roundtrip(self):
        register_hstore(self.conn)
        cur = self.conn.cursor()

        def ok(d):
            cur.execute("select %s", (d,))
            d1 = cur.fetchone()[0]
            self.assertEqual(len(d), len(d1))
            for k in d:
                self.assert_(k in d1, k)
                self.assertEqual(d[k], d1[k])

        ok({})
        ok({'a': 'b', 'c': None})

        ab = list(map(chr, range(32, 128)))
        ok(dict(zip(ab, ab)))
        ok({''.join(ab): ''.join(ab)})

        self.conn.set_client_encoding('latin1')
        ab = bytes(list(range(32, 127)) + list(range(160, 255))).decode('latin1')

        ok({''.join(ab): ''.join(ab)})
        ok(dict(zip(ab, ab)))

    @skip_if_no_hstore
    @restore_types
    def test_oid(self):
        cur = self.conn.cursor()
        cur.execute("select 'hstore'::regtype::oid")
        oid = cur.fetchone()[0]

        # Note: None as conn_or_cursor is just for testing: not public
        # interface and it may break in future.
        register_hstore(None, globally=True, oid=oid)
        cur.execute("select null::hstore, ''::hstore, 'a => b'::hstore")
        t = cur.fetchone()
        self.assert_(t[0] is None)
        self.assertEqual(t[1], {})
        self.assertEqual(t[2], {'a': 'b'})

    @skip_if_no_hstore
    @skip_before_postgres(8, 3)
    def test_roundtrip_array(self):
        register_hstore(self.conn)

        ds = [{}, {'a': 'b', 'c': None}]

        ab = list(map(chr, range(32, 128)))
        ds.append(dict(zip(ab, ab)))
        ds.append({''.join(ab): ''.join(ab)})

        self.conn.set_client_encoding('latin1')
        ab = bytes(list(range(32, 127)) + list(range(160, 255))).decode('latin1')

        ds.append({''.join(ab): ''.join(ab)})
        ds.append(dict(zip(ab, ab)))

        cur = self.conn.cursor()
        cur.execute("select %s", (ds,))
        ds1 = cur.fetchone()[0]
        self.assertEqual(ds, ds1)

    @skip_if_no_hstore
    @skip_before_postgres(8, 3)
    def test_array_cast(self):
        register_hstore(self.conn)
        cur = self.conn.cursor()
        cur.execute("select array['a=>1'::hstore, 'b=>2'::hstore];")
        a = cur.fetchone()[0]
        self.assertEqual(a, [{'a': '1'}, {'b': '2'}])

    @skip_if_no_hstore
    @restore_types
    def test_array_cast_oid(self):
        cur = self.conn.cursor()
        cur.execute("select 'hstore'::regtype::oid, 'hstore[]'::regtype::oid")
        oid, aoid = cur.fetchone()

        register_hstore(None, globally=True, oid=oid, array_oid=aoid)
        cur.execute("""
            select null::hstore, ''::hstore,
            'a => b'::hstore, '{a=>b}'::hstore[]""")
        t = cur.fetchone()
        self.assert_(t[0] is None)
        self.assertEqual(t[1], {})
        self.assertEqual(t[2], {'a': 'b'})
        self.assertEqual(t[3], [{'a': 'b'}])

    @skip_if_no_hstore
    def test_non_dbapi_connection(self):
        conn = self.connect(connection_factory=RealDictConnection)
        try:
            register_hstore(conn)
            curs = conn.cursor()
            curs.execute("select ''::hstore as x")
            self.assertEqual(curs.fetchone()['x'], {})
        finally:
            conn.close()

        conn = self.connect(connection_factory=RealDictConnection)
        try:
            curs = conn.cursor()
            register_hstore(curs)
            curs.execute("select ''::hstore as x")
            self.assertEqual(curs.fetchone()['x'], {})
        finally:
            conn.close()


def skip_if_no_composite(f):
    @wraps(f)
    @skip_if_crdb("composite")
    def skip_if_no_composite_(self):
        if self.conn.info.server_version < 80000:
            return self.skipTest(
                "server version %s doesn't support composite types"
                % self.conn.info.server_version)

        return f(self)

    return skip_if_no_composite_


class AdaptTypeTestCase(ConnectingTestCase):
    @skip_if_no_composite
    def test_none_in_record(self):
        curs = self.conn.cursor()
        s = curs.mogrify("SELECT %s;", [(42, None)])
        self.assertEqual(b"SELECT (42, NULL);", s)
        curs.execute("SELECT %s;", [(42, None)])
        d = curs.fetchone()[0]
        self.assertEqual("(42,)", d)

    def test_none_fast_path(self):
        # the None adapter is not actually invoked in regular adaptation

        class WonkyAdapter:
            def __init__(self, obj):
                pass

            def getquoted(self):
                return "NOPE!"

        curs = self.conn.cursor()

        orig_adapter = ext.adapters[type(None), ext.ISQLQuote]
        try:
            ext.register_adapter(type(None), WonkyAdapter)
            self.assertEqual(ext.adapt(None).getquoted(), "NOPE!")

            s = curs.mogrify("SELECT %s;", (None,))
            self.assertEqual(b"SELECT NULL;", s)

        finally:
            ext.register_adapter(type(None), orig_adapter)

    def test_tokenization(self):
        def ok(s, v):
            self.assertEqual(CompositeCaster.tokenize(s), v)

        ok("(,)", [None, None])
        ok('(,"")', [None, ''])
        ok('(hello,,10.234,2010-11-11)', ['hello', None, '10.234', '2010-11-11'])
        ok('(10,"""")', ['10', '"'])
        ok('(10,",")', ['10', ','])
        ok(r'(10,"\\")', ['10', '\\'])
        ok(r'''(10,"\\',""")''', ['10', '''\\',"'''])
        ok('(10,"(20,""(30,40)"")")', ['10', '(20,"(30,40)")'])
        ok('(10,"(20,""(30,""""(40,50)"""")"")")', ['10', '(20,"(30,""(40,50)"")")'])
        ok('(,"(,""(a\nb\tc)"")")', [None, '(,"(a\nb\tc)")'])
        ok('(\x01,\x02,\x03,\x04,\x05,\x06,\x07,\x08,"\t","\n","\x0b",'
           '"\x0c","\r",\x0e,\x0f,\x10,\x11,\x12,\x13,\x14,\x15,\x16,'
           '\x17,\x18,\x19,\x1a,\x1b,\x1c,\x1d,\x1e,\x1f," ",!,"""",#,'
           '$,%,&,\',"(",")",*,+,",",-,.,/,0,1,2,3,4,5,6,7,8,9,:,;,<,=,>,?,'
           '@,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,[,"\\\\",],'
           '^,_,`,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,{,|,},'
           '~,\x7f)',
           list(map(chr, range(1, 128))))
        ok('(,"\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f'
           '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !'
           '""#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\\\]'
           '^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f")',
           [None, ''.join(map(chr, range(1, 128)))])

    @skip_if_no_composite
    def test_cast_composite(self):
        oid = self._create_type("type_isd",
            [('anint', 'integer'), ('astring', 'text'), ('adate', 'date')])

        t = psycopg2.extras.register_composite("type_isd", self.conn)
        self.assertEqual(t.name, 'type_isd')
        self.assertEqual(t.schema, 'public')
        self.assertEqual(t.oid, oid)
        self.assert_(issubclass(t.type, tuple))
        self.assertEqual(t.attnames, ['anint', 'astring', 'adate'])
        self.assertEqual(t.atttypes, [23, 25, 1082])

        curs = self.conn.cursor()
        r = (10, 'hello', date(2011, 1, 2))
        curs.execute("select %s::type_isd;", (r,))
        v = curs.fetchone()[0]
        self.assert_(isinstance(v, t.type))
        self.assertEqual(v[0], 10)
        self.assertEqual(v[1], "hello")
        self.assertEqual(v[2], date(2011, 1, 2))
        self.assert_(t.type is not tuple)
        self.assertEqual(v.anint, 10)
        self.assertEqual(v.astring, "hello")
        self.assertEqual(v.adate, date(2011, 1, 2))

    @skip_if_no_composite
    def test_empty_string(self):
        # issue #141
        self._create_type("type_ss", [('s1', 'text'), ('s2', 'text')])
        curs = self.conn.cursor()
        psycopg2.extras.register_composite("type_ss", curs)

        def ok(t):
            curs.execute("select %s::type_ss", (t,))
            rv = curs.fetchone()[0]
            self.assertEqual(t, rv)

        ok(('a', 'b'))
        ok(('a', ''))
        ok(('', 'b'))
        ok(('a', None))
        ok((None, 'b'))
        ok(('', ''))
        ok((None, None))

    @skip_if_no_composite
    def test_cast_nested(self):
        self._create_type("type_is",
            [("anint", "integer"), ("astring", "text")])
        self._create_type("type_r_dt",
            [("adate", "date"), ("apair", "type_is")])
        self._create_type("type_r_ft",
            [("afloat", "float8"), ("anotherpair", "type_r_dt")])

        psycopg2.extras.register_composite("type_is", self.conn)
        psycopg2.extras.register_composite("type_r_dt", self.conn)
        psycopg2.extras.register_composite("type_r_ft", self.conn)

        curs = self.conn.cursor()
        r = (0.25, (date(2011, 1, 2), (42, "hello")))
        curs.execute("select %s::type_r_ft;", (r,))
        v = curs.fetchone()[0]

        self.assertEqual(r, v)
        self.assertEqual(v.anotherpair.apair.astring, "hello")

    @skip_if_no_composite
    def test_register_on_cursor(self):
        self._create_type("type_ii", [("a", "integer"), ("b", "integer")])

        curs1 = self.conn.cursor()
        curs2 = self.conn.cursor()
        psycopg2.extras.register_composite("type_ii", curs1)
        curs1.execute("select (1,2)::type_ii")
        self.assertEqual(curs1.fetchone()[0], (1, 2))
        curs2.execute("select (1,2)::type_ii")
        self.assertEqual(curs2.fetchone()[0], "(1,2)")

    @skip_if_no_composite
    def test_register_on_connection(self):
        self._create_type("type_ii", [("a", "integer"), ("b", "integer")])

        conn1 = self.connect()
        conn2 = self.connect()
        try:
            psycopg2.extras.register_composite("type_ii", conn1)
            curs1 = conn1.cursor()
            curs2 = conn2.cursor()
            curs1.execute("select (1,2)::type_ii")
            self.assertEqual(curs1.fetchone()[0], (1, 2))
            curs2.execute("select (1,2)::type_ii")
            self.assertEqual(curs2.fetchone()[0], "(1,2)")
        finally:
            conn1.close()
            conn2.close()

    @skip_if_no_composite
    @restore_types
    def test_register_globally(self):
        self._create_type("type_ii", [("a", "integer"), ("b", "integer")])

        conn1 = self.connect()
        conn2 = self.connect()
        try:
            psycopg2.extras.register_composite("type_ii", conn1, globally=True)
            curs1 = conn1.cursor()
            curs2 = conn2.cursor()
            curs1.execute("select (1,2)::type_ii")
            self.assertEqual(curs1.fetchone()[0], (1, 2))
            curs2.execute("select (1,2)::type_ii")
            self.assertEqual(curs2.fetchone()[0], (1, 2))

        finally:
            conn1.close()
            conn2.close()

    @skip_if_no_composite
    def test_composite_namespace(self):
        curs = self.conn.cursor()
        curs.execute("""
            select nspname from pg_namespace
            where nspname = 'typens';
            """)
        if not curs.fetchone():
            curs.execute("create schema typens;")
            self.conn.commit()

        self._create_type("typens.typens_ii",
            [("a", "integer"), ("b", "integer")])
        t = psycopg2.extras.register_composite(
            "typens.typens_ii", self.conn)
        self.assertEqual(t.schema, 'typens')
        curs.execute("select (4,8)::typens.typens_ii")
        self.assertEqual(curs.fetchone()[0], (4, 8))

    @skip_if_no_composite
    def test_composite_namespace_path(self):
        curs = self.conn.cursor()
        curs.execute("""
            select nspname from pg_namespace
            where nspname = 'typens';
            """)
        if not curs.fetchone():
            curs.execute("create schema typens;")
            self.conn.commit()

        self._create_type("typens.typensp_ii",
            [("a", "integer"), ("b", "integer")])
        curs.execute("set search_path=typens,public")
        t = psycopg2.extras.register_composite(
            "typensp_ii", self.conn)
        self.assertEqual(t.schema, 'typens')
        curs.execute("select (4,8)::typensp_ii")
        self.assertEqual(curs.fetchone()[0], (4, 8))

    @skip_if_no_composite
    def test_composite_weird_name(self):
        curs = self.conn.cursor()
        curs.execute("""
            select nspname from pg_namespace
            where nspname = 'qux.quux';
            """)
        if not curs.fetchone():
            curs.execute('create schema "qux.quux";')

        self._create_type('"qux.quux"."foo.bar"',
            [("a", "integer"), ("b", "integer")])
        t = psycopg2.extras.register_composite(
            '"qux.quux"."foo.bar"', self.conn)
        self.assertEqual(t.name, 'foo.bar')
        self.assertEqual(t.schema, 'qux.quux')
        curs.execute('select (4,8)::"qux.quux"."foo.bar"')
        self.assertEqual(curs.fetchone()[0], (4, 8))

    @skip_if_no_composite
    def test_composite_not_found(self):

        self.assertRaises(
            psycopg2.ProgrammingError, psycopg2.extras.register_composite,
            "nosuchtype", self.conn)
        self.assertEqual(self.conn.status, ext.STATUS_READY)

        cur = self.conn.cursor()
        cur.execute("select 1")
        self.assertRaises(
            psycopg2.ProgrammingError, psycopg2.extras.register_composite,
            "nosuchtype", self.conn)

        self.assertEqual(self.conn.status, ext.STATUS_IN_TRANSACTION)

        self.conn.rollback()
        self.conn.autocommit = True
        self.assertRaises(
            psycopg2.ProgrammingError, psycopg2.extras.register_composite,
            "nosuchtype", self.conn)
        self.assertEqual(self.conn.status, ext.STATUS_READY)

    @skip_if_no_composite
    @skip_before_postgres(8, 4)
    def test_composite_array(self):
        self._create_type("type_isd",
            [('anint', 'integer'), ('astring', 'text'), ('adate', 'date')])

        t = psycopg2.extras.register_composite("type_isd", self.conn)

        curs = self.conn.cursor()
        r1 = (10, 'hello', date(2011, 1, 2))
        r2 = (20, 'world', date(2011, 1, 3))
        curs.execute("select %s::type_isd[];", ([r1, r2],))
        v = curs.fetchone()[0]
        self.assertEqual(len(v), 2)
        self.assert_(isinstance(v[0], t.type))
        self.assertEqual(v[0][0], 10)
        self.assertEqual(v[0][1], "hello")
        self.assertEqual(v[0][2], date(2011, 1, 2))
        self.assert_(isinstance(v[1], t.type))
        self.assertEqual(v[1][0], 20)
        self.assertEqual(v[1][1], "world")
        self.assertEqual(v[1][2], date(2011, 1, 3))

    @skip_if_no_composite
    def test_wrong_schema(self):
        oid = self._create_type("type_ii", [("a", "integer"), ("b", "integer")])
        c = CompositeCaster('type_ii', oid, [('a', 23), ('b', 23), ('c', 23)])
        curs = self.conn.cursor()
        psycopg2.extensions.register_type(c.typecaster, curs)
        curs.execute("select (1,2)::type_ii")
        self.assertRaises(psycopg2.DataError, curs.fetchone)

    @slow
    @skip_if_no_composite
    @skip_before_postgres(8, 4)
    def test_from_tables(self):
        curs = self.conn.cursor()
        curs.execute("""create table ctest1 (
            id integer primary key,
            temp int,
            label varchar
        );""")

        curs.execute("""alter table ctest1 drop temp;""")

        curs.execute("""create table ctest2 (
            id serial primary key,
            label varchar,
            test_id integer references ctest1(id)
        );""")

        curs.execute("""insert into ctest1 (id, label) values
                (1, 'test1'),
                (2, 'test2');""")
        curs.execute("""insert into ctest2 (label, test_id) values
                ('testa', 1),
                ('testb', 1),
                ('testc', 2),
                ('testd', 2);""")

        psycopg2.extras.register_composite("ctest1", curs)
        psycopg2.extras.register_composite("ctest2", curs)

        curs.execute("""
            select ctest1, array_agg(ctest2) as test2s
            from (
                select ctest1, ctest2
                from ctest1 inner join ctest2 on ctest1.id = ctest2.test_id
                order by ctest1.id, ctest2.label
            ) x group by ctest1;""")

        r = curs.fetchone()
        self.assertEqual(r[0], (1, 'test1'))
        self.assertEqual(r[1], [(1, 'testa', 1), (2, 'testb', 1)])
        r = curs.fetchone()
        self.assertEqual(r[0], (2, 'test2'))
        self.assertEqual(r[1], [(3, 'testc', 2), (4, 'testd', 2)])

    @skip_if_no_composite
    def test_non_dbapi_connection(self):
        self._create_type("type_ii", [("a", "integer"), ("b", "integer")])

        conn = self.connect(connection_factory=RealDictConnection)
        try:
            register_composite('type_ii', conn)
            curs = conn.cursor()
            curs.execute("select '(1,2)'::type_ii as x")
            self.assertEqual(curs.fetchone()['x'], (1, 2))
        finally:
            conn.close()

        conn = self.connect(connection_factory=RealDictConnection)
        try:
            curs = conn.cursor()
            register_composite('type_ii', conn)
            curs.execute("select '(1,2)'::type_ii as x")
            self.assertEqual(curs.fetchone()['x'], (1, 2))
        finally:
            conn.close()

    @skip_if_no_composite
    def test_subclass(self):
        oid = self._create_type("type_isd",
            [('anint', 'integer'), ('astring', 'text'), ('adate', 'date')])

        class DictComposite(CompositeCaster):
            def make(self, values):
                return dict(zip(self.attnames, values))

        t = register_composite('type_isd', self.conn, factory=DictComposite)

        self.assertEqual(t.name, 'type_isd')
        self.assertEqual(t.oid, oid)

        curs = self.conn.cursor()
        r = (10, 'hello', date(2011, 1, 2))
        curs.execute("select %s::type_isd;", (r,))
        v = curs.fetchone()[0]
        self.assert_(isinstance(v, dict))
        self.assertEqual(v['anint'], 10)
        self.assertEqual(v['astring'], "hello")
        self.assertEqual(v['adate'], date(2011, 1, 2))

    def _create_type(self, name, fields):
        curs = self.conn.cursor()
        try:
            curs.execute("savepoint x")
            curs.execute(f"drop type {name} cascade;")
        except psycopg2.ProgrammingError:
            curs.execute("rollback to savepoint x")

        curs.execute("create type {} as ({});".format(name,
            ", ".join(["%s %s" % p for p in fields])))

        curs.execute("SELECT %s::regtype::oid", (name, ))
        oid = curs.fetchone()[0]
        self.conn.commit()
        return oid


def skip_if_no_json_type(f):
    """Skip a test if PostgreSQL json type is not available"""
    @wraps(f)
    def skip_if_no_json_type_(self):
        curs = self.conn.cursor()
        curs.execute("select oid from pg_type where typname = 'json'")
        if not curs.fetchone():
            return self.skipTest("json not available in test database")

        return f(self)

    return skip_if_no_json_type_


@skip_if_crdb("json")
class JsonTestCase(ConnectingTestCase):
    def test_adapt(self):
        objs = [None, "te'xt", 123, 123.45,
            '\xe0\u20ac', ['a', 100], {'a': 100}]

        curs = self.conn.cursor()
        for obj in enumerate(objs):
            self.assertQuotedEqual(curs.mogrify("%s", (Json(obj),)),
                psycopg2.extensions.QuotedString(json.dumps(obj)).getquoted())

    def test_adapt_dumps(self):
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)

        curs = self.conn.cursor()
        obj = Decimal('123.45')

        def dumps(obj):
            return json.dumps(obj, cls=DecimalEncoder)
        self.assertQuotedEqual(curs.mogrify("%s", (Json(obj, dumps=dumps),)),
            b"'123.45'")

    def test_adapt_subclass(self):
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)

        class MyJson(Json):
            def dumps(self, obj):
                return json.dumps(obj, cls=DecimalEncoder)

        curs = self.conn.cursor()
        obj = Decimal('123.45')
        self.assertQuotedEqual(curs.mogrify("%s", (MyJson(obj),)), b"'123.45'")

    @restore_types
    def test_register_on_dict(self):
        psycopg2.extensions.register_adapter(dict, Json)

        curs = self.conn.cursor()
        obj = {'a': 123}
        self.assertQuotedEqual(
            curs.mogrify("%s", (obj,)), b"""'{"a": 123}'""")

    def test_type_not_available(self):
        curs = self.conn.cursor()
        curs.execute("select oid from pg_type where typname = 'json'")
        if curs.fetchone():
            return self.skipTest("json available in test database")

        self.assertRaises(psycopg2.ProgrammingError,
            psycopg2.extras.register_json, self.conn)

    @skip_before_postgres(9, 2)
    def test_default_cast(self):
        curs = self.conn.cursor()

        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None})

        curs.execute("""select array['{"a": 100.0, "b": null}']::json[]""")
        self.assertEqual(curs.fetchone()[0], [{'a': 100.0, 'b': None}])

    @skip_if_no_json_type
    def test_register_on_connection(self):
        psycopg2.extras.register_json(self.conn)
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None})

    @skip_if_no_json_type
    def test_register_on_cursor(self):
        curs = self.conn.cursor()
        psycopg2.extras.register_json(curs)
        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None})

    @skip_if_no_json_type
    @restore_types
    def test_register_globally(self):
        new, newa = psycopg2.extras.register_json(self.conn, globally=True)
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None})

    @skip_if_no_json_type
    def test_loads(self):
        json = psycopg2.extras.json

        def loads(s):
            return json.loads(s, parse_float=Decimal)
        psycopg2.extras.register_json(self.conn, loads=loads)
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data['a'], Decimal))
        self.assertEqual(data['a'], Decimal('100.0'))

    @skip_if_no_json_type
    @restore_types
    def test_no_conn_curs(self):
        oid, array_oid = _get_json_oids(self.conn)

        def loads(s):
            return psycopg2.extras.json.loads(s, parse_float=Decimal)

        new, newa = psycopg2.extras.register_json(
            loads=loads, oid=oid, array_oid=array_oid)
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data['a'], Decimal))
        self.assertEqual(data['a'], Decimal('100.0'))

    @skip_before_postgres(9, 2)
    def test_register_default(self):
        curs = self.conn.cursor()

        def loads(s):
            return psycopg2.extras.json.loads(s, parse_float=Decimal)
        psycopg2.extras.register_default_json(curs, loads=loads)

        curs.execute("""select '{"a": 100.0, "b": null}'::json""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data['a'], Decimal))
        self.assertEqual(data['a'], Decimal('100.0'))

        curs.execute("""select array['{"a": 100.0, "b": null}']::json[]""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data[0]['a'], Decimal))
        self.assertEqual(data[0]['a'], Decimal('100.0'))

    @skip_if_no_json_type
    def test_null(self):
        psycopg2.extras.register_json(self.conn)
        curs = self.conn.cursor()
        curs.execute("""select NULL::json""")
        self.assertEqual(curs.fetchone()[0], None)
        curs.execute("""select NULL::json[]""")
        self.assertEqual(curs.fetchone()[0], None)

    def test_no_array_oid(self):
        curs = self.conn.cursor()
        t1, t2 = psycopg2.extras.register_json(curs, oid=25)
        self.assertEqual(t1.values[0], 25)
        self.assertEqual(t2, None)

        curs.execute("""select '{"a": 100.0, "b": null}'::text""")
        data = curs.fetchone()[0]
        self.assertEqual(data['a'], 100)
        self.assertEqual(data['b'], None)

    def test_str(self):
        snowman = "\u2603"
        obj = {'a': [1, 2, snowman]}
        j = psycopg2.extensions.adapt(psycopg2.extras.Json(obj))
        s = str(j)
        self.assert_(isinstance(s, str))
        # no pesky b's
        self.assert_(s.startswith("'"))
        self.assert_(s.endswith("'"))

    @skip_before_postgres(8, 2)
    def test_scs(self):
        cnn_on = self.connect(options="-c standard_conforming_strings=on")
        cur_on = cnn_on.cursor()
        self.assertEqual(
            cur_on.mogrify("%s", [psycopg2.extras.Json({'a': '"'})]),
            b'\'{"a": "\\""}\'')

        cnn_off = self.connect(options="-c standard_conforming_strings=off")
        cur_off = cnn_off.cursor()
        self.assertEqual(
            cur_off.mogrify("%s", [psycopg2.extras.Json({'a': '"'})]),
            b'E\'{"a": "\\\\""}\'')

        self.assertEqual(
            cur_on.mogrify("%s", [psycopg2.extras.Json({'a': '"'})]),
            b'\'{"a": "\\""}\'')


def skip_if_no_jsonb_type(f):
    return skip_before_postgres(9, 4)(f)


@skip_if_no_jsonb_type
class JsonbTestCase(ConnectingTestCase):
    @staticmethod
    def myloads(s):
        rv = json.loads(s)
        rv['test'] = 1
        return rv

    def test_default_cast(self):
        curs = self.conn.cursor()

        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None})

        if crdb_version(self.conn) is None:
            curs.execute("""select array['{"a": 100.0, "b": null}']::jsonb[]""")
            self.assertEqual(curs.fetchone()[0], [{'a': 100.0, 'b': None}])

    def test_register_on_connection(self):
        psycopg2.extras.register_json(self.conn, loads=self.myloads, name='jsonb')
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None, 'test': 1})

    def test_register_on_cursor(self):
        curs = self.conn.cursor()
        psycopg2.extras.register_json(curs, loads=self.myloads, name='jsonb')
        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None, 'test': 1})

    @restore_types
    def test_register_globally(self):
        new, newa = psycopg2.extras.register_json(self.conn,
            loads=self.myloads, globally=True, name='jsonb')
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        self.assertEqual(curs.fetchone()[0], {'a': 100.0, 'b': None, 'test': 1})

    def test_loads(self):
        json = psycopg2.extras.json

        def loads(s):
            return json.loads(s, parse_float=Decimal)

        psycopg2.extras.register_json(self.conn, loads=loads, name='jsonb')
        curs = self.conn.cursor()
        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data['a'], Decimal))
        self.assertEqual(data['a'], Decimal('100.0'))
        # sure we are not mangling json too?
        if crdb_version(self.conn) is None:
            curs.execute("""select '{"a": 100.0, "b": null}'::json""")
            data = curs.fetchone()[0]
            self.assert_(isinstance(data['a'], float))
            self.assertEqual(data['a'], 100.0)

    def test_register_default(self):
        curs = self.conn.cursor()

        def loads(s):
            return psycopg2.extras.json.loads(s, parse_float=Decimal)

        psycopg2.extras.register_default_jsonb(curs, loads=loads)

        curs.execute("""select '{"a": 100.0, "b": null}'::jsonb""")
        data = curs.fetchone()[0]
        self.assert_(isinstance(data['a'], Decimal))
        self.assertEqual(data['a'], Decimal('100.0'))

        if crdb_version(self.conn) is None:
            curs.execute("""select array['{"a": 100.0, "b": null}']::jsonb[]""")
            data = curs.fetchone()[0]
            self.assert_(isinstance(data[0]['a'], Decimal))
            self.assertEqual(data[0]['a'], Decimal('100.0'))

    def test_null(self):
        curs = self.conn.cursor()
        curs.execute("""select NULL::jsonb""")
        self.assertEqual(curs.fetchone()[0], None)
        if crdb_version(self.conn) is None:
            curs.execute("""select NULL::jsonb[]""")
            self.assertEqual(curs.fetchone()[0], None)


class RangeTestCase(unittest.TestCase):
    def test_noparam(self):
        r = Range()

        self.assert_(not r.isempty)
        self.assertEqual(r.lower, None)
        self.assertEqual(r.upper, None)
        self.assert_(r.lower_inf)
        self.assert_(r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_empty(self):
        r = Range(empty=True)

        self.assert_(r.isempty)
        self.assertEqual(r.lower, None)
        self.assertEqual(r.upper, None)
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_nobounds(self):
        r = Range(10, 20)
        self.assertEqual(r.lower, 10)
        self.assertEqual(r.upper, 20)
        self.assert_(not r.isempty)
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_bounds(self):
        for bounds, lower_inc, upper_inc in [
                ('[)', True, False),
                ('(]', False, True),
                ('()', False, False),
                ('[]', True, True)]:
            r = Range(10, 20, bounds)
            self.assertEqual(r.lower, 10)
            self.assertEqual(r.upper, 20)
            self.assert_(not r.isempty)
            self.assert_(not r.lower_inf)
            self.assert_(not r.upper_inf)
            self.assertEqual(r.lower_inc, lower_inc)
            self.assertEqual(r.upper_inc, upper_inc)

    def test_keywords(self):
        r = Range(upper=20)
        self.assertEqual(r.lower, None)
        self.assertEqual(r.upper, 20)
        self.assert_(not r.isempty)
        self.assert_(r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

        r = Range(lower=10, bounds='(]')
        self.assertEqual(r.lower, 10)
        self.assertEqual(r.upper, None)
        self.assert_(not r.isempty)
        self.assert_(not r.lower_inf)
        self.assert_(r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_bad_bounds(self):
        self.assertRaises(ValueError, Range, bounds='(')
        self.assertRaises(ValueError, Range, bounds='[}')

    def test_in(self):
        r = Range(empty=True)
        self.assert_(10 not in r)

        r = Range()
        self.assert_(10 in r)

        r = Range(lower=10, bounds='[)')
        self.assert_(9 not in r)
        self.assert_(10 in r)
        self.assert_(11 in r)

        r = Range(lower=10, bounds='()')
        self.assert_(9 not in r)
        self.assert_(10 not in r)
        self.assert_(11 in r)

        r = Range(upper=20, bounds='()')
        self.assert_(19 in r)
        self.assert_(20 not in r)
        self.assert_(21 not in r)

        r = Range(upper=20, bounds='(]')
        self.assert_(19 in r)
        self.assert_(20 in r)
        self.assert_(21 not in r)

        r = Range(10, 20)
        self.assert_(9 not in r)
        self.assert_(10 in r)
        self.assert_(11 in r)
        self.assert_(19 in r)
        self.assert_(20 not in r)
        self.assert_(21 not in r)

        r = Range(10, 20, '(]')
        self.assert_(9 not in r)
        self.assert_(10 not in r)
        self.assert_(11 in r)
        self.assert_(19 in r)
        self.assert_(20 in r)
        self.assert_(21 not in r)

        r = Range(20, 10)
        self.assert_(9 not in r)
        self.assert_(10 not in r)
        self.assert_(11 not in r)
        self.assert_(19 not in r)
        self.assert_(20 not in r)
        self.assert_(21 not in r)

    def test_nonzero(self):
        self.assert_(Range())
        self.assert_(Range(10, 20))
        self.assert_(not Range(empty=True))

    def test_eq_hash(self):
        def assert_equal(r1, r2):
            self.assert_(r1 == r2)
            self.assert_(hash(r1) == hash(r2))

        assert_equal(Range(empty=True), Range(empty=True))
        assert_equal(Range(), Range())
        assert_equal(Range(10, None), Range(10, None))
        assert_equal(Range(10, 20), Range(10, 20))
        assert_equal(Range(10, 20), Range(10, 20, '[)'))
        assert_equal(Range(10, 20, '[]'), Range(10, 20, '[]'))

        def assert_not_equal(r1, r2):
            self.assert_(r1 != r2)
            self.assert_(hash(r1) != hash(r2))

        assert_not_equal(Range(10, 20), Range(10, 21))
        assert_not_equal(Range(10, 20), Range(11, 20))
        assert_not_equal(Range(10, 20, '[)'), Range(10, 20, '[]'))

    def test_eq_wrong_type(self):
        self.assertNotEqual(Range(10, 20), ())

    def test_eq_subclass(self):
        class IntRange(NumericRange):
            pass

        class PositiveIntRange(IntRange):
            pass

        self.assertEqual(Range(10, 20), IntRange(10, 20))
        self.assertEqual(PositiveIntRange(10, 20), IntRange(10, 20))

    # as the postgres docs describe for the server-side stuff,
    # ordering is rather arbitrary, but will remain stable
    # and consistent.

    def test_lt_ordering(self):
        self.assert_(Range(empty=True) < Range(0, 4))
        self.assert_(not Range(1, 2) < Range(0, 4))
        self.assert_(Range(0, 4) < Range(1, 2))
        self.assert_(not Range(1, 2) < Range())
        self.assert_(Range() < Range(1, 2))
        self.assert_(not Range(1) < Range(upper=1))
        self.assert_(not Range() < Range())
        self.assert_(not Range(empty=True) < Range(empty=True))
        self.assert_(not Range(1, 2) < Range(1, 2))
        with raises_typeerror():
            self.assert_(1 < Range(1, 2))
        with raises_typeerror():
            self.assert_(not Range(1, 2) < 1)

    def test_gt_ordering(self):
        self.assert_(not Range(empty=True) > Range(0, 4))
        self.assert_(Range(1, 2) > Range(0, 4))
        self.assert_(not Range(0, 4) > Range(1, 2))
        self.assert_(Range(1, 2) > Range())
        self.assert_(not Range() > Range(1, 2))
        self.assert_(Range(1) > Range(upper=1))
        self.assert_(not Range() > Range())
        self.assert_(not Range(empty=True) > Range(empty=True))
        self.assert_(not Range(1, 2) > Range(1, 2))
        with raises_typeerror():
            self.assert_(not 1 > Range(1, 2))
        with raises_typeerror():
            self.assert_(Range(1, 2) > 1)

    def test_le_ordering(self):
        self.assert_(Range(empty=True) <= Range(0, 4))
        self.assert_(not Range(1, 2) <= Range(0, 4))
        self.assert_(Range(0, 4) <= Range(1, 2))
        self.assert_(not Range(1, 2) <= Range())
        self.assert_(Range() <= Range(1, 2))
        self.assert_(not Range(1) <= Range(upper=1))
        self.assert_(Range() <= Range())
        self.assert_(Range(empty=True) <= Range(empty=True))
        self.assert_(Range(1, 2) <= Range(1, 2))
        with raises_typeerror():
            self.assert_(1 <= Range(1, 2))
        with raises_typeerror():
            self.assert_(not Range(1, 2) <= 1)

    def test_ge_ordering(self):
        self.assert_(not Range(empty=True) >= Range(0, 4))
        self.assert_(Range(1, 2) >= Range(0, 4))
        self.assert_(not Range(0, 4) >= Range(1, 2))
        self.assert_(Range(1, 2) >= Range())
        self.assert_(not Range() >= Range(1, 2))
        self.assert_(Range(1) >= Range(upper=1))
        self.assert_(Range() >= Range())
        self.assert_(Range(empty=True) >= Range(empty=True))
        self.assert_(Range(1, 2) >= Range(1, 2))
        with raises_typeerror():
            self.assert_(not 1 >= Range(1, 2))
        with raises_typeerror():
            self.assert_(Range(1, 2) >= 1)

    def test_pickling(self):
        r = Range(0, 4)
        self.assertEqual(loads(dumps(r)), r)

    def test_str(self):
        '''
        Range types should have a short and readable ``str`` implementation.

        Using ``repr`` for all string conversions can be very unreadable for
        longer types like ``DateTimeTZRange``.
        '''

        # Using the "u" prefix to make sure we have the proper return types in
        # Python2
        expected = [
            '(0, 4)',
            '[0, 4]',
            '(0, 4]',
            '[0, 4)',
            'empty',
        ]
        results = []

        for bounds in ('()', '[]', '(]', '[)'):
            r = Range(0, 4, bounds=bounds)
            results.append(str(r))

        r = Range(empty=True)
        results.append(str(r))
        self.assertEqual(results, expected)

    def test_str_datetime(self):
        '''
        Date-Time ranges should return a human-readable string as well on
        string conversion.
        '''
        tz = timezone(timedelta(minutes=-5 * 60), "EST")
        r = DateTimeTZRange(datetime(2010, 1, 1, tzinfo=tz),
                            datetime(2011, 1, 1, tzinfo=tz))
        expected = '[2010-01-01 00:00:00-05:00, 2011-01-01 00:00:00-05:00)'
        result = str(r)
        self.assertEqual(result, expected)


@skip_if_crdb("range")
@skip_before_postgres(9, 2, "range not supported before postgres 9.2")
class RangeCasterTestCase(ConnectingTestCase):

    builtin_ranges = ('int4range', 'int8range', 'numrange',
        'daterange', 'tsrange', 'tstzrange')

    def test_cast_null(self):
        cur = self.conn.cursor()
        for type in self.builtin_ranges:
            cur.execute(f"select NULL::{type}")
            r = cur.fetchone()[0]
            self.assertEqual(r, None)

    def test_cast_empty(self):
        cur = self.conn.cursor()
        for type in self.builtin_ranges:
            cur.execute(f"select 'empty'::{type}")
            r = cur.fetchone()[0]
            self.assert_(isinstance(r, Range), type)
            self.assert_(r.isempty)

    def test_cast_inf(self):
        cur = self.conn.cursor()
        for type in self.builtin_ranges:
            cur.execute(f"select '(,)'::{type}")
            r = cur.fetchone()[0]
            self.assert_(isinstance(r, Range), type)
            self.assert_(not r.isempty)
            self.assert_(r.lower_inf)
            self.assert_(r.upper_inf)

    def test_cast_numbers(self):
        cur = self.conn.cursor()
        for type in ('int4range', 'int8range'):
            cur.execute(f"select '(10,20)'::{type}")
            r = cur.fetchone()[0]
            self.assert_(isinstance(r, NumericRange))
            self.assert_(not r.isempty)
            self.assertEqual(r.lower, 11)
            self.assertEqual(r.upper, 20)
            self.assert_(not r.lower_inf)
            self.assert_(not r.upper_inf)
            self.assert_(r.lower_inc)
            self.assert_(not r.upper_inc)

        cur.execute("select '(10.2,20.6)'::numrange")
        r = cur.fetchone()[0]
        self.assert_(isinstance(r, NumericRange))
        self.assert_(not r.isempty)
        self.assertEqual(r.lower, Decimal('10.2'))
        self.assertEqual(r.upper, Decimal('20.6'))
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_cast_date(self):
        cur = self.conn.cursor()
        cur.execute("select '(2000-01-01,2012-12-31)'::daterange")
        r = cur.fetchone()[0]
        self.assert_(isinstance(r, DateRange))
        self.assert_(not r.isempty)
        self.assertEqual(r.lower, date(2000, 1, 2))
        self.assertEqual(r.upper, date(2012, 12, 31))
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_cast_timestamp(self):
        cur = self.conn.cursor()
        ts1 = datetime(2000, 1, 1)
        ts2 = datetime(2000, 12, 31, 23, 59, 59, 999)
        cur.execute("select tsrange(%s, %s, '()')", (ts1, ts2))
        r = cur.fetchone()[0]
        self.assert_(isinstance(r, DateTimeRange))
        self.assert_(not r.isempty)
        self.assertEqual(r.lower, ts1)
        self.assertEqual(r.upper, ts2)
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(not r.lower_inc)
        self.assert_(not r.upper_inc)

    def test_cast_timestamptz(self):
        cur = self.conn.cursor()
        ts1 = datetime(2000, 1, 1, tzinfo=timezone(timedelta(minutes=600)))
        ts2 = datetime(2000, 12, 31, 23, 59, 59, 999,
                       tzinfo=timezone(timedelta(minutes=600)))
        cur.execute("select tstzrange(%s, %s, '[]')", (ts1, ts2))
        r = cur.fetchone()[0]
        self.assert_(isinstance(r, DateTimeTZRange))
        self.assert_(not r.isempty)
        self.assertEqual(r.lower, ts1)
        self.assertEqual(r.upper, ts2)
        self.assert_(not r.lower_inf)
        self.assert_(not r.upper_inf)
        self.assert_(r.lower_inc)
        self.assert_(r.upper_inc)

    def test_adapt_number_range(self):
        cur = self.conn.cursor()

        r = NumericRange(empty=True)
        cur.execute("select %s::int4range", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange))
        self.assert_(r1.isempty)

        r = NumericRange(10, 20)
        cur.execute("select %s::int8range", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange))
        self.assertEqual(r1.lower, 10)
        self.assertEqual(r1.upper, 20)
        self.assert_(r1.lower_inc)
        self.assert_(not r1.upper_inc)

        r = NumericRange(Decimal('10.2'), Decimal('20.5'), '(]')
        cur.execute("select %s::numrange", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange))
        self.assertEqual(r1.lower, Decimal('10.2'))
        self.assertEqual(r1.upper, Decimal('20.5'))
        self.assert_(not r1.lower_inc)
        self.assert_(r1.upper_inc)

    def test_adapt_numeric_range(self):
        cur = self.conn.cursor()

        r = NumericRange(empty=True)
        cur.execute("select %s::int4range", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange), r1)
        self.assert_(r1.isempty)

        r = NumericRange(10, 20)
        cur.execute("select %s::int8range", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange))
        self.assertEqual(r1.lower, 10)
        self.assertEqual(r1.upper, 20)
        self.assert_(r1.lower_inc)
        self.assert_(not r1.upper_inc)

        r = NumericRange(Decimal('10.2'), Decimal('20.5'), '(]')
        cur.execute("select %s::numrange", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, NumericRange))
        self.assertEqual(r1.lower, Decimal('10.2'))
        self.assertEqual(r1.upper, Decimal('20.5'))
        self.assert_(not r1.lower_inc)
        self.assert_(r1.upper_inc)

    def test_adapt_date_range(self):
        cur = self.conn.cursor()

        d1 = date(2012, 1, 1)
        d2 = date(2012, 12, 31)
        r = DateRange(d1, d2)
        cur.execute("select %s", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, DateRange))
        self.assertEqual(r1.lower, d1)
        self.assertEqual(r1.upper, d2)
        self.assert_(r1.lower_inc)
        self.assert_(not r1.upper_inc)

        r = DateTimeRange(empty=True)
        cur.execute("select %s", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, DateTimeRange))
        self.assert_(r1.isempty)

        ts1 = datetime(2000, 1, 1, tzinfo=timezone(timedelta(minutes=600)))
        ts2 = datetime(2000, 12, 31, 23, 59, 59, 999,
                       tzinfo=timezone(timedelta(minutes=600)))
        r = DateTimeTZRange(ts1, ts2, '(]')
        cur.execute("select %s", (r,))
        r1 = cur.fetchone()[0]
        self.assert_(isinstance(r1, DateTimeTZRange))
        self.assertEqual(r1.lower, ts1)
        self.assertEqual(r1.upper, ts2)
        self.assert_(not r1.lower_inc)
        self.assert_(r1.upper_inc)

    @restore_types
    def test_register_range_adapter(self):
        cur = self.conn.cursor()
        cur.execute("create type textrange as range (subtype=text)")
        rc = register_range('textrange', 'TextRange', cur)

        TextRange = rc.range
        self.assert_(issubclass(TextRange, Range))
        self.assertEqual(TextRange.__name__, 'TextRange')

        r = TextRange('a', 'b', '(]')
        cur.execute("select %s", (r,))
        r1 = cur.fetchone()[0]
        self.assertEqual(r1.lower, 'a')
        self.assertEqual(r1.upper, 'b')
        self.assert_(not r1.lower_inc)
        self.assert_(r1.upper_inc)

        cur.execute("select %s", ([r, r, r],))
        rs = cur.fetchone()[0]
        self.assertEqual(len(rs), 3)
        for r1 in rs:
            self.assertEqual(r1.lower, 'a')
            self.assertEqual(r1.upper, 'b')
            self.assert_(not r1.lower_inc)
            self.assert_(r1.upper_inc)

    def test_range_escaping(self):
        cur = self.conn.cursor()
        cur.execute("create type textrange as range (subtype=text)")
        rc = register_range('textrange', 'TextRange', cur)

        TextRange = rc.range
        cur.execute("""
            create table rangetest (
                id integer primary key,
                range textrange)""")

        bounds = ['[)', '(]', '()', '[]']
        ranges = [TextRange(low, up, bounds[i % 4])
            for i, (low, up) in enumerate(zip(
                [None] + list(map(chr, range(1, 128))),
                list(map(chr, range(1, 128))) + [None],
            ))]
        ranges.append(TextRange())
        ranges.append(TextRange(empty=True))

        errs = 0
        for i, r in enumerate(ranges):
            # not all the ranges make sense:
            # fun fact: select ascii('#') < ascii('$'), '#' < '$'
            # yelds... t, f! At least in en_GB.UTF-8 collation.
            # which seems suggesting a supremacy of the pound on the dollar.
            # So some of these ranges will fail to insert. Be prepared but...
            try:
                cur.execute("""
                    savepoint x;
                    insert into rangetest (id, range) values (%s, %s);
                    """, (i, r))
            except psycopg2.DataError:
                errs += 1
                cur.execute("rollback to savepoint x;")

        # ...not too many errors! in the above collate there are 17 errors:
        # assume in other collates we won't find more than 30
        self.assert_(errs < 30,
            "too many collate errors. Is the test working?")

        cur.execute("select id, range from rangetest order by id")
        for i, r in cur:
            self.assertEqual(ranges[i].lower, r.lower)
            self.assertEqual(ranges[i].upper, r.upper)
            self.assertEqual(ranges[i].lower_inc, r.lower_inc)
            self.assertEqual(ranges[i].upper_inc, r.upper_inc)
            self.assertEqual(ranges[i].lower_inf, r.lower_inf)
            self.assertEqual(ranges[i].upper_inf, r.upper_inf)

        # clear the adapters to allow precise count by scripts/refcounter.py
        del ext.adapters[TextRange, ext.ISQLQuote]

    def test_range_not_found(self):
        cur = self.conn.cursor()
        self.assertRaises(psycopg2.ProgrammingError,
            register_range, 'nosuchrange', 'FailRange', cur)
        self.assertEqual(self.conn.status, ext.STATUS_READY)

        cur.execute("select 1")
        self.assertRaises(psycopg2.ProgrammingError,
            register_range, 'nosuchrange', 'FailRange', cur)

        self.assertEqual(self.conn.status, ext.STATUS_IN_TRANSACTION)

        self.conn.rollback()
        self.conn.autocommit = True
        self.assertRaises(psycopg2.ProgrammingError,
            register_range, 'nosuchrange', 'FailRange', cur)

    @restore_types
    def test_schema_range(self):
        cur = self.conn.cursor()
        cur.execute("create schema rs")
        cur.execute("create type r1 as range (subtype=text)")
        cur.execute("create type r2 as range (subtype=text)")
        cur.execute("create type rs.r2 as range (subtype=text)")
        cur.execute("create type rs.r3 as range (subtype=text)")
        cur.execute("savepoint x")

        register_range('r1', 'r1', cur)
        ra2 = register_range('r2', 'r2', cur)
        rars2 = register_range('rs.r2', 'r2', cur)
        rars3 = register_range('rs.r3', 'r3', cur)

        self.assertNotEqual(
            ra2.typecaster.values[0],
            rars2.typecaster.values[0])

        self.assertRaises(psycopg2.ProgrammingError,
            register_range, 'r3', 'FailRange', cur)
        cur.execute("rollback to savepoint x;")

        self.assertRaises(psycopg2.ProgrammingError,
            register_range, 'rs.r1', 'FailRange', cur)
        cur.execute("rollback to savepoint x;")

        cur2 = self.conn.cursor()
        cur2.execute("set local search_path to rs,public")
        ra3 = register_range('r3', 'r3', cur2)
        self.assertEqual(ra3.typecaster.values[0], rars3.typecaster.values[0])

    @skip_if_no_composite
    def test_rang_weird_name(self):
        cur = self.conn.cursor()
        cur.execute("""
            select nspname from pg_namespace
            where nspname = 'qux.quux';
            """)
        if not cur.fetchone():
            cur.execute('create schema "qux.quux";')

        cur.execute('create type "qux.quux"."foo.range" as range (subtype=text)')
        r = psycopg2.extras.register_range(
            '"qux.quux"."foo.range"', "foorange", cur)
        cur.execute('''select '[a,z]'::"qux.quux"."foo.range"''')
        self.assertEqual(cur.fetchone()[0], r.range('a', 'z', '[]'))


def test_suite():
    return unittest.TestLoader().loadTestsFromName(__name__)


if __name__ == "__main__":
    unittest.main()
