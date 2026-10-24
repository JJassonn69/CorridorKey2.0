import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .common import canonical


def ident(prefix):
    return prefix + '_' + uuid.uuid4().hex


class Problem(Exception):
    def __init__(self, status, code, detail=''):
        self.status, self.code, self.detail = status, code, detail
        super().__init__(code)


class Store:
    """All access is serialized by the controller lock; transactions never span I/O."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'state.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT PRIMARY KEY,owner TEXT,doc TEXT);
          CREATE INDEX IF NOT EXISTS by_owner ON records(kind,owner);
          CREATE TABLE IF NOT EXISTS tokens(hash TEXT PRIMARY KEY,owner TEXT,expires REAL);
          CREATE TABLE IF NOT EXISTS keys(owner TEXT,path TEXT,key TEXT,body TEXT,kind TEXT,id TEXT,expires REAL,
            PRIMARY KEY(owner,path,key));
        ''')

    def put(self, kind, owner, doc):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',
                            (kind, doc['id'], owner, canonical(doc).decode()))

    def get(self, kind, id, owner=None):
        row = self.db.execute('SELECT owner,doc FROM records WHERE kind=? AND id=?', (kind, id)).fetchone()
        if not row or (owner is not None and row[0] != owner):
            raise Problem(404, 'not_found')
        return json.loads(row[1])

    def all(self, kind):
        return [(owner, json.loads(doc)) for owner, doc in self.db.execute('SELECT owner,doc FROM records WHERE kind=? ORDER BY rowid', (kind,))]

    def authenticate(self, token):
        row = self.db.execute('SELECT owner,expires FROM tokens WHERE hash=?',
                              (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not row or row[1] <= time.time():
            raise Problem(401, 'unauthorized')
        return row[0]

    def replay(self, owner, path, key, body):
        if not key or len(key) > 128:
            raise Problem(400, 'idempotency_key_required')
        h = hashlib.sha256(canonical(body)).hexdigest()
        row = self.db.execute('SELECT body,kind,id,expires FROM keys WHERE owner=? AND path=? AND key=?',
                              (owner, path, key)).fetchone()
        if row:
            if row[0] != h:
                raise Problem(409, 'idempotency_conflict')
            if row[3] <= time.time():
                raise Problem(410, 'idempotency_expired')
            doc = self.get(row[1], row[2], owner)
            if doc['state'] in ('expired', 'deleted', 'deleting'):
                raise Problem(410, 'resource_gone')
            return doc

    def remember(self, owner, path, key, body, kind, doc):
        # The mutation and its key commit together. put() is not called in here.
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',
                            (kind, doc['id'], owner, canonical(doc).decode()))
            self.db.execute('INSERT INTO keys VALUES(?,?,?,?,?,?,?)',
                            (owner, path, key, hashlib.sha256(canonical(body)).hexdigest(), kind, doc['id'], time.time()+604800))
