"""Wheat water footprint ML pipeline.

Modules follow the layout documented in the repository README:
    io_utils, grids            - shared helpers
    wf, crop_calendar,
    climate, soil              - one per raw data source
    integration                - spatial join into the master dataset
    features, preprocess       - feature engineering, cleaning, split
    feature_selection          - Pearson + XGBoost -> 20 features
    clustering, models         - K-means / hierarchical, cluster-wise AdaBoost
    evaluation, plots          - metrics and figures
"""
__version__ = "0.1.0"
