#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_listdetail
----------------------------------

Tests for `swmmtoolbox` module.
"""
import shlex
import subprocess
import os
import six
import csv

from unittest import TestCase
from pandas.util.testing import assert_frame_equal
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas as pd

from swmmtoolbox import swmmtoolbox


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture outputA
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        return bytes(out, 'utf-8')
    except:
        return out


class TestListdetail(TestCase):
    def setUp(self):
        listdetail_node_fname = os.path.join('tests', 'listdetail_node.csv')
        self.listdetail_node = pd.read_csv(listdetail_node_fname,
                                           sep=',',
                                           skipinitialspace=True)

        listdetail_link_fname = os.path.join('tests', 'listdetail_link.csv')
        self.listdetail_link = pd.read_csv(listdetail_link_fname,
                                           sep=',',
                                           skipinitialspace=True)

    def test_listdetail_node(self):
        out = swmmtoolbox.listdetail(os.path.join('tests', 'frutal.out'),
                                     'node')
        self.maxDiff = None
        assert_frame_equal(out, self.listdetail_node)

    def test_listdetail_link(self):
        out = swmmtoolbox.listdetail(os.path.join('tests', 'frutal.out'),
                                     'link')
        self.maxDiff = None
        assert_frame_equal(out, self.listdetail_link)
