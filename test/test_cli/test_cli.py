import sys
import unittest
from unittest import mock

import tractography as tg


class TestCLI(unittest.TestCase):

    def test_help(self):
        """Test the help command of the CLI"""
        #test_main()


@mock.patch("sys.argv", return_value=["tractography", "-h"])
def test_main(*args, **kwargs):
    tg.cli.main()
