

import unittest
from Backtesting import Weights


class TestWeights(unittest.TestCase):

    def test_myMethod_happy_path(self):
        obj = classA(...)
        result = obj.myMethod(input_value)
        self.assertEqual(result, expected_value)

    def test_myMethod_edge_case(self):
        obj = classA(...)
        result = obj.myMethod(edge_input)
        self.assertEqual(result, edge_expected)

if __name__ == "__main__":
    unittest.main()
