"""Desktop smoke check: exercise demo, fit and inverse calculation, then close."""
import calibration
import tkinter as tk
from tkinter import ttk, messagebox

original_tk = tk.Tk
errors = []


def fail(title, message):
    errors.append(f'{title}: {message}')


messagebox.showerror = fail


def test_root():
    root = original_tk()
    root.withdraw()

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    def exercise():
        try:
            widgets = list(descendants(root))
            buttons = {w.cget('text'): w for w in widgets if isinstance(w, ttk.Button)}
            buttons['Demo data'].invoke()
            buttons['Fit 4PL'].invoke()
            reports = [w.get('1.0', 'end') for w in widgets if isinstance(w, tk.Text)]
            assert any('R² =' in text for text in reports), reports
            entry = next(w for w in widgets if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox))
            entry.insert(0, '1.0')
            combo = next(w for w in widgets if isinstance(w, ttk.Combobox))
            combo.set('y → x')
            buttons['Calculate'].invoke()
            assert any('1 →' in w.cget('text') for w in widgets if isinstance(w, ttk.Label))
            assert not errors, errors
            print('GUI verified: demo, fit, plot, report, inverse calculation.')
        except Exception as exc:
            errors.append(str(exc))
        finally:
            root.destroy()

    root.after(200, exercise)
    return root


tk.Tk = test_root
calibration.main()
if errors:
    raise RuntimeError(errors)
