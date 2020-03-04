#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import datasets
import sys

class dataprocessing():

    def __init__(self, config):
        self.config = config.dataprocessing_params
        self.config_full = config
        print("Loading dataset", config.dataprocessing_params.dataset, "...")
        self.dataset = getattr(datasets, config.dataprocessing_params.dataset)(self.config_full)



    def run(self):
        self.dataset.run()
        self.dataset.clean()
        self.dataset.pack()
