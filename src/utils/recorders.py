import numpy as np

class Recorder:
    def __init__(self):
        self.temp_records = {}
        self.aggregated_records ={}

    def record_statistics(self, obj):
        for k, v in obj.report_statistics().items():
            self._append_to_map(self.temp_records, k, v)
            
    def _append_to_map(self, m, k, v):
        if k not in m:
            m[k] = []

        m[k].append(v)

    def _empty_temp_records(self):
        self.temp_records = {}

    def _aggregate_data(self, key, agg_method):
        if agg_method == 'avg':
            self._append_to_map(
                self.aggregated_records, 
                f'average_{key}', 
                np.average(self.temp_records[key])
            )
        elif agg_method == 'max':
            self._append_to_map(
                self.aggregated_records, 
                f'max_{key}', 
                np.max(self.temp_records[key])
            )
        elif agg_method == 'any':
            self._append_to_map(
                self.aggregated_records, 
                key, 
                self.temp_records[key][0]
            )

    def aggregate_statistics(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, str):
                self._aggregate_data(k, v)
            elif isinstance(v, list):
                for agg in v:
                    self._aggregate_data(k, agg)

        self._empty_temp_records()

    def report_statistics(self):
        return {k: np.array(v) for k, v in self.aggregated_records.items()}