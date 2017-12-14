# -*- coding: utf-8 -*-

from unittest import TestCase

import os


class Test_PRUNE(TestCase):

    def test_running(self):
        os.system('python src/main.py')
