# -*- coding: utf-8 -*-
"""Tests for DoS prevention mechanisms in sqlparse."""

import pytest
import sqlparse
from sqlparse import sql


def _count_flatten_calls(monkeypatch):
    """Count TokenList.flatten() calls (each one walks a whole subtree)."""
    calls = [0]
    original_flatten = sql.TokenList.flatten

    def counting_flatten(self):
        calls[0] += 1
        return original_flatten(self)

    monkeypatch.setattr(sql.TokenList, 'flatten', counting_flatten)
    return calls


def _assert_group_values_consistent(tlist):
    """Every group's cached value must equal its flattened text."""
    for token in tlist.tokens:
        if token.is_group:
            _assert_group_values_consistent(token)
    assert tlist.value == u''.join(t.value for t in tlist.flatten())


class TestDoSPrevention:
    """Test cases to ensure sqlparse is protected against DoS attacks."""

    def test_extended_group_value_not_recomputed(self, monkeypatch):
        """Extending a group must not re-flatten the whole group.

        CVE-2026-54284: group_tokens(extend=True) recomputed the group value
        from the complete subtree on every extension, so a long
        comma-separated list cost O(n**2).
        """
        n = 2000
        sql_str = 'SELECT {0} FROM table1'.format(
            ', '.join('col{0}'.format(i) for i in range(n)))

        calls = _count_flatten_calls(monkeypatch)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: one subtree walk per comma, roughly n**2 / 2 calls.
        assert calls[0] < n
        assert isinstance(stmt.tokens[2], sql.IdentifierList)
        assert stmt.value == sql_str
        _assert_group_values_consistent(stmt)

    def test_nested_group_creation_does_not_flatten_subtree(
            self, monkeypatch):
        """Creating a group must not re-flatten its nested child groups.

        CVE-2026-54284: TokenList.__init__ derived its value from str(self),
        walking every nested group again, so left-nested groups such as
        a long operator chain cost O(n**2).
        """
        n = 50
        sql_str = 'SELECT ' + ' + '.join(['1'] * n)

        calls = _count_flatten_calls(monkeypatch)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: each new Operation walks all previous ones,
        # roughly n**2 / 2 calls.
        assert calls[0] < n
        assert isinstance(stmt.tokens[2], sql.Operation)
        assert stmt.value == sql_str
        _assert_group_values_consistent(stmt)

    def test_comment_only_input_not_rescanned(self, monkeypatch):
        """group_comments must not rescan the tail per comment token.

        CVE-2026-71491: on comment-only input group_comments searched the
        remaining tokens for a terminator once per comment token, so a
        statement of n single-line comments cost O(n**2).
        """
        n = 500
        sql_str = '-- c\n' * n

        calls = [0]
        original = sql.TokenList.token_not_matching

        def counting(self, funcs, idx):
            calls[0] += 1
            return original(self, funcs, idx)

        monkeypatch.setattr(sql.TokenList, 'token_not_matching', counting)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: one tail scan per comment token, i.e. n calls.
        assert calls[0] < 10
        assert stmt.value == sql_str
        assert sqlparse.format(sql_str, strip_comments=True).strip() == ''

    def test_comments_before_statement_still_grouped(self):
        stmt = sqlparse.parse('-- a\n-- b\nSELECT 1')[0]
        assert isinstance(stmt.tokens[0], sql.Comment)
        assert stmt.tokens[0].value == '-- a\n-- b\n'
