# -*- coding: utf-8 -*-

from unittest import TestCase

import os


class TestAttriRank(TestCase):

    def test_running(self):
        os.system('python src/main.py')
        os.remove('graph.embeddings')
