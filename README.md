# Fourier Analysis Studio V3.3.1

Patch release over V3.3.

## Additions

- Added a cumulative reconstruction chart that layers components until they match the reconstructed/original curve.
- Kept the component window explorer for 1-day, 1-week, and 1-month views.
- Added an option to show cumulative boundary lines.
- Preserved 2-decimal on-screen formatting and full CSV precision.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```


## V4 notes

- Explicit **Run Fourier analysis** button to avoid re-running expensive analysis on every interaction.
- Safer error metrics and grouped error tables for empty or fully-masked series.
- Extra regression tests for empty-array and all-NaN edge cases.


## Packaging note

Pytest run during packaging reported issues:

```
[32m.[0m[32m.[0m[32m.[0m[31mF[0m[31mF[0m[32m.[0m[32m.[0m[32m.[0m[32m.[0m[31mF[0m[31mF[0m[32m.[0m[32m.[0m[32m.[0m[31m                                                           [100%][0m
=================================== FAILURES ===================================
[31m[1m_____________________ test_component_window_figure_builds ______________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_component_window_figure_builds[39;49;00m():[90m[39;49;00m
>       [94mfrom[39;49;00m[90m [39;49;00m[04m[96msrc[39;49;00m[04m[96m.[39;49;00m[04m[96mplotting[39;49;00m[90m [39;49;00m[94mimport[39;49;00m build_component_window_figure[90m[39;49;00m

[1m[31mtests/test_decomposition.py[0m:64: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    [0m[94mfrom[39;49;00m[90m [39;49;00m[04m[96m__future__[39;49;00m[90m [39;49;00m[94mimport[39;49;00m annotations[90m[39;49;00m
    [90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mnumpy[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mnp[39;49;00m[90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mpandas[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mpd[39;49;00m[90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mplotly[39;49;00m[04m[96m.[39;49;00m[04m[96mexpress[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mpx[39;49;00m[90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mplotly[39;49;00m[04m[96m.[39;49;00m[04m[96mgraph_objects[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mgo[39;49;00m[90m[39;49;00m
>   [94mimport[39;49;00m[90m [39;49;00m[04m[96mstreamlit[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mst[39;49;00m[90m[39;49;00m
[1m[31mE   ModuleNotFoundError: No module named 'streamlit'[0m

[1m[31msrc/plotting.py[0m:7: ModuleNotFoundError
[31m[1m___________________ test_cumulative_components_figure_builds ___________________[0m

    [0m[94mdef[39;49;00m[90m [39;49;00m[92mtest_cumulative_components_figure_builds[39;49;00m():[90m[39;49;00m
>       [94mfrom[39;49;00m[90m [39;49;00m[04m[96msrc[39;49;00m[04m[96m.[39;49;00m[04m[96mplotting[39;49;00m[90m [39;49;00m[94mimport[39;49;00m build_cumulative_components_figure[90m[39;49;00m

[1m[31mtests/test_decomposition.py[0m:81: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    [0m[94mfrom[39;49;00m[90m [39;49;00m[04m[96m__future__[39;49;00m[90m [39;49;00m[94mimport[39;49;00m annotations[90m[39;49;00m
    [90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mnumpy[39;49;00m[90m [39;49;00m[94mas[39;49;00m[90m [39;49;00m[04m[96mnp[39;49;00m[90m[39;49;00m
    [94mimport[39;49;00m[90m [39;49;00m[04m[96mpandas[39;49;00m[90m 
```


## V4.1 hotfix

- Fixed `error_metrics()` for empty arrays so tests on empty/all-NaN cases pass correctly.


## V4.2 features

- Analysis scope selector: **Global** or **Annual**.
- Annual mode computes one Fourier analysis per eligible year and exports a multi-year CSV summary.
- Two-step workflow retained: preview first, then click **Run Fourier analysis**.


## V4.2.1 hotfix

- Fixed Annual mode summary field mismatch for top period hours.
- Added explicit unique `key` values to all `st.plotly_chart(...)` calls to avoid `StreamlitDuplicateElementId`.


## V4.2.2 hotfix

- Added explicit unique `key` values to all `st.download_button(...)` calls to avoid `StreamlitDuplicateElementId` in download widgets.


## V4.3 features

- Annual mode now shows a year coverage / eligibility table and explains why only one year may be selectable.
- Added automatic display-only rounding for nearly flat low-frequency / trend components.
- Kept unique keys for Plotly charts and download buttons to avoid Streamlit duplicate element ID errors.


## V4.4 features

- Added initial period selection: full series, custom date range, or single year.
- Added pre-Fourier diagnostics and preview charts for the selected analysis window.
- Fourier analysis now runs on the selected period instead of always using the full dataset.


## V4.4.1 features

- Added residual diagnostics tab with residual time series, distribution, autocorrelation, heatmap, and top residual events.
- Added performance hardening with cache limits, signature-based invalidation, lighter previews, and downsampled plots for large datasets.
- Added collapsible glossary in the initial screen.


## V4.4.2 features

- Consolidated navigation into 4 main tabs: Overview, Frequency & Reconstruction, Components & Residuals, Exports.
- Optimized sidebar with grouped sections and advanced settings inside expanders.
- Preserved residual diagnostics, initial period selection, and large-series downsampling.
