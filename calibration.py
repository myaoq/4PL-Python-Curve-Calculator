"""4PL calibration: import fit_4pl, or run this file for the desktop app."""
from pathlib import Path
import sys

# Optional project-local dependencies installed for this computer.
if (Path(__file__).parent / 'deps').is_dir():
    sys.path.insert(0, str(Path(__file__).parent / 'deps'))

import csv
import io
import re
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def four_pl(x, a, b, c, d):
    """y = d + (a-d)/(1+(x/c)**b); b,c > 0, x >= 0.

    a is the response at zero; d is the infinite-x asymptote.
    Both increasing (d>a) and decreasing (d<a) curves are supported.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError('x must contain finite, nonnegative values.')
    if not np.all(np.isfinite([a, b, c, d])) or b <= 0 or c <= 0:
        raise ValueError('Parameters must be finite; b and c must be positive.')
    with np.errstate(divide='ignore'):
        return a + (d-a) * expit(b * (np.log(x) - np.log(c)))


@dataclass
class Calibration:
    a: float
    b: float
    c: float
    d: float
    x: np.ndarray
    y: np.ndarray
    r_squared: float
    rmse: float
    warnings: list

    def predict_y(self, x):
        return four_pl(x, self.a, self.b, self.c, self.d)

    def predict_x(self, y):
        """Inverse, including x=0 at y=a; reject the unreachable y=d."""
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or self.d == self.a:
            raise ValueError('y must be finite and the curve must not be flat.')
        fraction = (y-self.a)/(self.d-self.a)
        if np.any((fraction < 0) | (fraction >= 1)):
            raise ValueError(
                f'y must lie between {self.a:.10g} (included, x=0) and '
                f'{self.d:.10g} (excluded, reached only as x tends to infinity).')
        with np.errstate(divide='ignore', over='ignore'):
            result = np.exp(np.log(self.c) +
                            (np.log(fraction)-np.log1p(-fraction))/self.b)
        if not np.all(np.isfinite(result)):
            raise ValueError('The inverse exceeds numerical limits near the asymptote.')
        return result

    def report(self):
        return (
            '4PL: y = d + (a - d) / (1 + (x / c)^b)\n'
            f'a = {self.a:.12g}    b = {self.b:.12g}\n'
            f'c = {self.c:.12g}    d = {self.d:.12g}\n\n'
            f'y = ({self.d:.12g}) + ({self.a-self.d:.12g}) / '
            f'(1 + (x / ({self.c:.12g}))^({self.b:.12g}))\n'
            'Inverse: x = c * ((y - a) / (d - y))^(1 / b)\n'
            f'x = ({self.c:.12g}) * ((y - ({self.a:.12g})) / '
            f'(({self.d:.12g}) - y))^(1 / ({self.b:.12g}))\n\n'
            f'R² = {self.r_squared:.6g}    RMSE = {self.rmse:.6g}\n'
            f'Calibration x range: {self.x.min():.10g} to {self.x.max():.10g}\n'
            'Unweighted least squares; every replicate is fitted separately.\n'
            + '\n'.join('Warning: ' + w for w in self.warnings))


def fit_4pl(x, y):
    """Fit using scaled responses and multiple bounded least-squares starts."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError('x and y must be one-dimensional arrays of equal length.')
    if x.size < 5 or np.unique(x).size < 4:
        raise ValueError('Use at least 5 data pairs with at least 4 distinct x values; 6–8 levels are preferable.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x < 0):
        raise ValueError('All data must be finite numbers and x must be nonnegative.')
    scale, offset = float(np.ptp(y)), float(np.min(y))
    if scale <= np.finfo(float).eps * max(1, float(np.max(np.abs(y)))):
        raise ValueError('Responses are constant or too close to constant to fit.')
    z = (y-offset)/scale
    positive = x[x > 0]
    log_min, log_max = float(np.log(positive.min())), float(np.log(positive.max()))
    with np.errstate(divide='ignore'):
        log_x = np.log(x)

    def residual(p):
        a, d, log_c, log_b = p
        return a + (d-a)*expit(np.exp(log_b)*(log_x-log_c)) - z

    # Bounds avoid numerically degenerate slopes/midpoints; boundary fits are flagged.
    lower = [-np.inf, -np.inf, max(-700, log_min-14), np.log(0.01)]
    upper = [np.inf, np.inf, min(700, log_max+14), np.log(100)]
    candidates = []
    for a0, d0 in [(0, 1), (1, 0)]:
        for log_c0 in np.linspace(log_min, log_max, 3):
            for b0 in [0.5, 2.0]:
                result = least_squares(residual, [a0, d0, log_c0, np.log(b0)],
                                       bounds=(lower, upper), max_nfev=2500,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if result.success and np.all(np.isfinite(result.fun)):
                    candidates.append(result)
    if not candidates:
        raise ValueError('The fit did not converge. Check the data and calibration range.')
    best = min(candidates, key=lambda r: np.sum(r.fun**2))
    az, dz, log_c, log_b = best.x
    a, d, c, b = az*scale+offset, dz*scale+offset, np.exp(log_c), np.exp(log_b)
    sse_scaled = float(np.sum(best.fun**2))
    r2 = 1-sse_scaled/float(np.sum((z-z.mean())**2))
    warnings = []
    if np.linalg.cond(best.jac) > 1e8:
        warnings.append('Parameters are poorly determined; add standards covering both plateaus.')
    if np.any(best.active_mask):
        warnings.append('A parameter reached an optimization bound; inspect the fit carefully.')
    if not x.min() < c < x.max():
        warnings.append('The fitted midpoint is outside the calibration range.')
    coverage = expit(b*(np.log(positive)-np.log(c)))
    if (x.min() > 0 and coverage.min() > 0.1) or coverage.max() < 0.9:
        warnings.append('Standards do not cover both fitted plateaus; asymptotes may be uncertain.')
    if r2 < 0.95:
        warnings.append('R² is below 0.95; inspect residuals. R² alone does not validate a calibration.')
    return Calibration(float(a), float(b), float(c), float(d), x.copy(), y.copy(),
                       r2, scale*np.sqrt(sse_scaled/x.size), warnings)


def parse_pairs(text):
    """Read two columns: comma, semicolon, tab or whitespace; optional header."""
    rows = []
    for number, line in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        if not line.strip():
            continue
        delimiter = next((d for d in ['\t', ';', ','] if d in line), None)
        cells = next(csv.reader([line], delimiter=delimiter)) if delimiter else line.split()
        if len(cells) != 2:
            raise ValueError(f'Line {number}: expected exactly two columns (x and y).')
        try:
            pair = [float(v.strip()) for v in cells]
        except ValueError:
            # A header must have two textual names, not a partly malformed data row.
            def numeric(s):
                try:
                    float(s)
                    return True
                except ValueError:
                    return False
            if not rows and all(not numeric(v) and re.search('[a-zA-Z]', v) for v in cells):
                if number == next(i for i, v in enumerate(text.lstrip('\ufeff').splitlines(), 1) if v.strip()):
                    continue
            raise ValueError(f'Line {number}: both cells must be numbers.') from None
        rows.append(pair)
    if not rows:
        raise ValueError('Paste or load calibration data first.')
    return np.asarray(rows, dtype=float).T


DEMO = 'x,y\n0,0.10\n0.1,0.104\n0.3,0.121\n1,0.210\n3,0.489\n10,1.05\n30,1.57\n100,1.89\n300,1.97\n'


def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title('4PL Calibration Curve Tool')
    root.geometry('1160x850')
    root.minsize(950, 700)
    fitted = None
    fitted_text = None
    top = ttk.Frame(root, padding=12)
    top.pack(fill='both', expand=True)
    left = ttk.Frame(top)
    left.pack(side='left', fill='y', padx=(0, 12))
    ttk.Label(left, text='Calibration standards', font=('Segoe UI', 14, 'bold')).pack(anchor='w')
    ttk.Label(left, text='Paste x and y columns from Excel.\nUse original x values, not log(x).').pack(anchor='w', pady=6)
    data = tk.Text(left, width=34, height=17, undo=True)
    data.pack(fill='x')
    buttons = ttk.Frame(left)
    buttons.pack(fill='x', pady=8)
    right = ttk.Frame(top)
    right.pack(side='left', fill='both', expand=True)
    figure = Figure(figsize=(6, 4), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, right)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    NavigationToolbar2Tk(canvas, right)
    report = tk.Text(right, height=16, wrap='word', font=('Consolas', 10))
    report.pack(fill='x')
    log_axis = tk.BooleanVar(value=True)

    def set_report(text):
        report.configure(state='normal')
        report.delete('1.0', 'end')
        report.insert('1.0', text)
        report.configure(state='disabled')

    def draw():
        ax.clear()
        if fitted is not None:
            x = fitted.x
            positive = x[x > 0]
            grid = np.geomspace(positive.min(), x.max(), 500)
            if x.min() == 0:
                grid = np.r_[0, np.geomspace(positive.min()/100, positive.min(), 60), grid]
            ax.scatter(x, fitted.y, label='Standards', color='#176B87', zorder=3)
            ax.plot(grid, fitted.predict_y(grid), label='4PL fit', color='#D97706')
            if log_axis.get():
                if x.min() == 0:
                    ax.set_xscale('symlog', linthresh=positive.min())
                    ax.set_xlabel('x (symlog: linear near zero, logarithmic above)')
                else:
                    ax.set_xscale('log')
                    ax.set_xlabel('x (log scale)')
            else:
                ax.set_xlabel('x')
            ax.legend()
        ax.set_ylabel('y (response)')
        ax.set_title('4-parameter logistic calibration')
        ax.grid(alpha=0.2)
        figure.tight_layout()
        canvas.draw()

    def load_csv():
        path = filedialog.askopenfilename(filetypes=[('CSV / text', '*.csv *.tsv *.txt'), ('All files', '*.*')])
        if path:
            try:
                text = Path(path).read_text(encoding='utf-8-sig')
                parse_pairs(text)
                data.delete('1.0', 'end')
                data.insert('1.0', text)
            except Exception as exc:
                messagebox.showerror('Cannot load data', str(exc))

    def demo():
        data.delete('1.0', 'end')
        data.insert('1.0', DEMO)

    def fit():
        nonlocal fitted, fitted_text
        fitted = None
        fitted_text = None
        output.configure(text='')
        set_report('Fitting…')
        root.update_idletasks()
        try:
            text = data.get('1.0', 'end').strip()
            fitted = fit_4pl(*parse_pairs(text))
            fitted_text = text
            set_report(fitted.report())
        except Exception as exc:
            set_report('No valid fit. Correct the data and fit again.')
            messagebox.showerror('Cannot fit curve', str(exc))
        draw()

    def require_fit():
        if fitted is None:
            raise ValueError('Fit your calibration data first.')
        if data.get('1.0', 'end').strip() != fitted_text:
            raise ValueError('Calibration data changed. Click Fit 4PL again.')

    ttk.Button(buttons, text='Load CSV', command=load_csv).pack(side='left')
    ttk.Button(buttons, text='Demo data', command=demo).pack(side='left', padx=4)
    ttk.Button(left, text='Fit 4PL', command=fit).pack(fill='x')
    ttk.Checkbutton(left, text='Logarithmic x-axis', variable=log_axis, command=draw).pack(anchor='w', pady=8)
    ttk.Separator(left).pack(fill='x', pady=12)
    ttk.Label(left, text='Calculate unknowns', font=('Segoe UI', 12, 'bold')).pack(anchor='w')
    mode = tk.StringVar(value='x → y')
    ttk.Combobox(left, textvariable=mode, values=['x → y', 'y → x'], state='readonly').pack(fill='x', pady=6)
    ttk.Label(left, text='Enter one or more values (comma separated):').pack(anchor='w')
    values = ttk.Entry(left)
    values.pack(fill='x', pady=6)
    output = ttk.Label(left, text='', wraplength=285, justify='left')

    def calculate():
        try:
            require_fit()
            inputs = [float(v) for v in re.split(r'[,;\s]+', values.get().strip()) if v]
            if not inputs:
                raise ValueError('Enter at least one value.')
            lines = []
            for v in inputs:
                try:
                    result = float(fitted.predict_y(v) if mode.get() == 'x → y' else fitted.predict_x(v))
                    x_value = v if mode.get() == 'x → y' else result
                    flag = ' [extrapolation]' if not fitted.x.min() <= x_value <= fitted.x.max() else ''
                    lines.append(f'{v:.8g} → {result:.10g}{flag}')
                except ValueError as exc:
                    lines.append(f'{v:.8g}: {exc}')
            output.configure(text='\n'.join(lines))
        except Exception as exc:
            output.configure(text='')
            messagebox.showerror('Cannot calculate', str(exc))

    ttk.Button(left, text='Calculate', command=calculate).pack(fill='x')
    output.pack(fill='x', pady=8)

    def save():
        try:
            require_fit()
            path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Fit report', '*.txt')])
            if path:
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerow(['x', 'observed_y', 'fitted_y', 'residual'])
                for x, y, prediction in zip(fitted.x, fitted.y, fitted.predict_y(fitted.x)):
                    writer.writerow([x, y, prediction, y-prediction])
                Path(path).write_text(fitted.report()+'\n\n'+buffer.getvalue(), encoding='utf-8')
        except Exception as exc:
            messagebox.showerror('Cannot save report', str(exc))

    ttk.Button(left, text='Save fit report', command=save).pack(fill='x', pady=8)
    set_report('Paste your standards and click Fit 4PL.\nOr click Demo data to try an example.')
    draw()
    root.mainloop()


if __name__ == '__main__':
    main()
