# -*- coding: utf-8 -*-
"""Tests for DoS prevention mechanisms in sqlparse."""

import pytest
import sqlparse
from sqlparse import keywords, lexer, sql
from sqlparse import tokens as T


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

    def test_lexer_has_no_rescanning_delimiter_regex(self):
        """No per-position lexer regex may scan ahead for a closing delimiter.

        CVE-2026-59893 (GHSA-prg7-hcfm-mfcr): dollar-quoted literals and
        multiline comments were matched with a lazy dot-all quantifier
        terminated by a backreference or a literal delimiter, applied at
        every text position by the lexer loop, so every unclosed opener
        re-scanned the rest of the input.
        """
        for rexmatch, _ in keywords.SQL_REGEX:
            pattern = rexmatch.__self__.pattern
            assert '[\\s\\S]*?' not in pattern, pattern

    @pytest.mark.parametrize('opener', ['$a{0}$x', '/* c{0}', '/*+ h{0}'])
    def test_unclosed_delimiters_resolved_once_per_opener(
            self, monkeypatch, opener):
        """Each unclosed opener costs one lookup, not a rescan of the tail.

        CVE-2026-59893: with many unique unclosed `$tag$` or `/*` openers
        the lexer re-scanned the remaining input once per opener, O(n**2).
        """
        n = 2000
        sql_str = ' '.join(opener.format(i) for i in range(n))

        calls = [0]
        original = keywords.DelimitedSpans.resolve

        def counting(self, pos):
            calls[0] += 1
            return original(self, pos)

        monkeypatch.setattr(keywords.DelimitedSpans, 'resolve', counting)
        stream = list(lexer.tokenize(sql_str))

        assert calls[0] == n
        assert u''.join(value for _, value in stream) == sql_str
        assert not any(ttype in (T.Literal, T.Comment.Multiline)
                       for ttype, _ in stream)

    def test_many_unclosed_openers_still_pair_with_closer(self):
        """Unclosed openers in between must not break delimiter pairing."""
        n = 2000
        body = ' '.join('/* c{0}'.format(i) for i in range(n)) + ' */'
        stream = list(lexer.tokenize(body + ' SELECT 1'))
        assert stream[0] == (T.Comment.Multiline, body)

        body = '$q$ {0} $q$'.format(
            ' '.join('$a{0}$x'.format(i) for i in range(n)))
        stream = list(lexer.tokenize(body + ' SELECT 1'))
        assert stream[0] == (T.Literal, body)
