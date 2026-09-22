import unittest
import calibration as cal
import numpy as np


class CalibrationTests(unittest.TestCase):
    def test_increasing_and_decreasing_recovery_and_inverse(self):
        x = np.r_[0, np.geomspace(0.01, 1000, 20)]
        for a, d in [(0.1, 2.5), (2.5, 0.1)]:
            y = cal.four_pl(x, a, 1.4, 7.5, d)
            fit = cal.fit_4pl(x, y)
            np.testing.assert_allclose([fit.a, fit.b, fit.c, fit.d], [a, 1.4, 7.5, d], rtol=1e-5)
            np.testing.assert_allclose(fit.predict_x(fit.predict_y(x)), x, rtol=1e-7, atol=1e-10)
            self.assertAlmostEqual(float(fit.predict_x(fit.a)), 0)
            for invalid in [fit.d, max(fit.a, fit.d)+1, min(fit.a, fit.d)-1, np.nan]:
                with self.assertRaises(ValueError):
                    fit.predict_x(invalid)

    def test_noisy_replicates(self):
        rng = np.random.default_rng(42)
        x = np.repeat(np.r_[0, np.geomspace(0.01, 1000, 12)], 3)
        y = cal.four_pl(x, 0.2, 1.2, 10, 3) + rng.normal(0, 0.02, x.size)
        fit = cal.fit_4pl(x, y)
        self.assertGreater(fit.r_squared, 0.99)
        np.testing.assert_allclose(fit.predict_x([1.0, 2.0]),
                                   10*((np.array([1.0, 2.0])-0.2)/(3-np.array([1.0, 2.0])))**(1/1.2), rtol=0.05)

    def test_invalid_input(self):
        cases = [([1, 2, 3], [1, 2, 3]),
                 ([0, 1, 2, 3, 4], [1]*5),
                 ([-1, 0, 1, 2, 3], [0, 1, 2, 3, 4]),
                 ([0, 1, 2, 3, 4], [0, 1, np.nan, 3, 4]),
                 ([1, 1, 1, 2, 2], [1, 2, 3, 4, 5])]
        for x, y in cases:
            with self.assertRaises(ValueError):
                cal.fit_4pl(x, y)

    def test_parser(self):
        for text in ['x,y\n0,1\n2,3', '\ufeffConcentration\tResponse\n0\t1\n2\t3', '0 1\n2 3', 'x;y\n0;1\n2;3']:
            np.testing.assert_array_equal(cal.parse_pairs(text), [[0, 2], [1, 3]])
        for text in ['', 'x,y\n1,bad', 'x,y\n1,2,3', '1,bad', 'x,y\nx,y']:
            with self.assertRaises(ValueError):
                cal.parse_pairs(text)


if __name__ == '__main__':
    unittest.main()
