# 4PL calibration tool

On this computer, double-click **Start 4PL Tool.cmd**.

1. Paste two columns from Excel: x (concentration) and y (response), or load a CSV. An optional header is allowed. Use decimal points, no thousands separators. Enter original concentrations, not log concentrations.
2. Click **Fit 4PL** to see the plot, numerical equation, parameters, R² and RMSE.
3. Select **x → y** or **y → x**, enter values separated by commas, and click **Calculate**.
4. Save the equation and calibration residuals with **Save fit report**. The plot toolbar saves the graph as an image.

Use **Demo data** to try the app without your own standards. Demo values are synthetic.

## Model

`y = d + (a - d) / (1 + (x / c)^b)`

- `a`: response at x = 0.
- `d`: response approached at very large x.
- `c`: positive midpoint concentration, where y = (a + d)/2.
- `b`: positive slope magnitude; d > a gives an increasing curve, d < a a decreasing curve.

Inverse: `x = c * ((y - a) / (d - y))^(1 / b)`.

Here `^` denotes exponentiation; use `**` in Python. Calculations use full precision, while displayed equations round parameters to 12 significant digits.

Input x values must be nonnegative. At least five observations and four distinct x values are required; preferably use 6–8 or more concentration levels spanning the transition and both plateaus. Repeat x values are allowed and each replicate gets equal weight. Zero concentration is supported; the log plot uses a linear region near zero when needed.

The inverse accepts y between the fitted asymptotes, including y=a (x=0), excluding y=d (no finite x). Values outside the calibration x range are marked as extrapolation. Estimates close to a plateau are sensitive to small response errors. A good R² does not establish that standards cover the plateaus or that inverse predictions are accurate; inspect residuals and fit warnings.

Fitting is unweighted nonlinear least squares with multiple starting points, normalized responses, and positive slope/midpoint parameters. Slope is bounded to 0.01–100; the midpoint can extend about six orders of magnitude beyond positive standards. Fits reaching numerical bounds or having an ill-conditioned Jacobian are flagged. Parameter uncertainty intervals and weighted fitting are not implemented.

Numerical solver reference: [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## Run on another computer

Install Python 3.10 or newer with Tkinter (included in the standard Windows Python installer), then run from this folder:

```sh
python -m pip install -r requirements.txt
python calibration.py
```

The optional `deps` folder contains packages for this computer's bundled Python. When moving the source to another computer, copy the source files without `deps` and install requirements there.

## Use in your own Python code

```python
from calibration import fit_4pl

x = [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
y = [0.10, 0.104, 0.121, 0.210, 0.489, 1.05, 1.57, 1.89, 1.97]
curve = fit_4pl(x, y)
print(curve.report())
print(curve.predict_y([2, 5, 20]))
print(curve.predict_x([0.5, 1.0, 1.5]))
```

The Python API validates domains but does not reject extrapolation; check predicted/input x against `curve.x.min()` and `curve.x.max()`. The desktop app labels extrapolation automatically.

Run verification with `python -m unittest -v test_calibration`.
