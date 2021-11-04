Changelog for Mantarray Waveform Analysis
=========================================


1.2.0 (2021-11-05)
------------------

- Updated all analysis to require units of time in microseconds.


1.1.1 (2021-09-30)
------------------

- Fixed new metrics from previous release.


1.1.0 (2021-08-31)
------------------

- Added the following metrics:

  - Ratio of amplitude of each peak to the that of the largest peak.
  - Time from X% width to peak.
  - Time from peak to X% width.


1.0.3 (2021-07-21)
------------------

- Added ability to invert peak detection for magnetic data.


1.0.1 (2021-07-15)
------------------

- Added ``metrics_to_create`` kwarg to ``data_metrics`` to specify which metrics to create. This kwarg can
  also be passed to Pipeline methods that get data metrics.


1.0.0 (2021-05-06)
------------------

- Removed ``filtered_data`` serialized data from ``PeakDectectionError`` subclassed exceptions and from ``find_twitch_indices`` function arguments for performance reasons.


0.9.0 (2021-04-26)
------------------

- Added Twitch Interval Irregularity metric to per twitch metrics sheet and aggregate metrics sheet.


0.8.1 (2021-04-23)
------------------

- Fixed the out of bounds error present in h5 file due to the fix in the  ``peak_detector`` which mitigated the TwoValleysInARow error.


0.8.0 (2021-04-21)
------------------

- Fixed the directionality of the twitch depending on whether the data metrics in the pipeline template are magnetic or force metrics.


0.7.1 (2021-04-01)
------------------

- Fixed TwoValleysInARowError present in h5 file due to ``peak_detector`` in which the second peak is taller than the second.


0.7.0 (2021-03-17)
------------------

- Updated calculations to convert magnetic data to Force


0.6.0 (2021-03-01)
------------------

- Added twitch contraction and relaxation velocity metrics to metric dicitonaries


0.5.11 (2021-02-17)
------------------

- Fixed TwoValleysInARowError present in h5 file due to ``peak_detector`` function


0.5.10 (2020-11-17)
------------------

- Updated dependencies to be compatible with Python 3.9


0.5.8 (2020-11-10)
------------------

- Fixed issue with peak detection on data with no detected valleys.


0.5.6 (2020-11-04)
------------------

- Fixed ``publish`` job.
- Publishing using github workflow and build environment.
- Fixed incorrect raising of TwoValleysInARowError.
- Fixed issue with two valleys incorrectly being found between peaks.


0.5.4 (2020-09-30)
------------------

- Fixed peak detection so it is less likely to report two peaks/valleys in a row.


0.5.3 (2020-09-15)
------------------

- Added TwoValleysInARowError.
- Fixed TwoPeaksInARowError reporting.


0.5.2 (2020-09-09)
------------------

- Added upload of source files to pypi for linux python3.7 download.


0.5.1 (2020-09-09)
------------------

- Added 30 Hz Butterworth Filter.


0.5.0 (2020-09-08)
------------------

- Added Twitch Frequency metric.
- Added peak detetection and metric calculation (for magnetic signal) to Pipeline.
- Created alias of load_raw_magnetic_data to become more agnostic to sensor type.


0.4.1 (2020-09-02)
------------------

- Added 30 Hz Low-Pass Bessel filter.
- Added small speed upgrade to cython compression code.


0.4.0 (2020-09-01)
------------------

- Refactored twitch width analysis so that it interpolates to find a point to use.
- Added aggregate statistic metrics for twitch widths.
- Refactored peak detection to be more robust.
- Cached the filter coefficients in PipelineTemplate to improve performance.


0.3.1 (2020-08-31)
------------------

- Added compression speed improvements.
- Fixed edge case in compression for horizontal line r squared.
