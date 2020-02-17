#! /usr/bin/env python
# -*- coding: utf-8 -*-
import json
numeric_type = {  # 直接分桶 特征名字:分桶区间
    'qk601': [0, 5000, 20000],
    'indinc': [0, 5000, 20000],
    'land_asset': [0, 20000, 100000],
    'total_asset': [0, 100000, 500000],
    'expense': [0, 50000, 200000],
    'fproperty': [0, 100000, 500000],
}

numeric_type_with_med = {  # 利用中位数分桶 特征名字:中位数加减的数目
    'fe601': 1000,
    'fe802': 5000,
    'fe903': 5000,
    'ff2': 5000,
}

numeric_type_json = json.dumps(numeric_type)
with open('numeric_type.json', 'w', encoding='UTF-8') as fw:
    fw.write(numeric_type_json)

numeric_type_with_med_json = json.dumps(numeric_type_with_med)
with open('numeric_type_with_med.json', 'w', encoding='UTF-8') as fw:
    fw.write(numeric_type_with_med_json)

