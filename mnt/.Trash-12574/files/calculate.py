# vim: fileencoding=utf-8 ai ts=4 sts=4 et sw=4
import argparse
from decimal import Decimal
import json
from os import path

import requests

from settings import *


FILE_PATH = path.join('/vagrant', 'api', 'v1')
HEADERS = {
    'x-api-app-authorization': APP_KEY,
    'x-requested-with': 'XMLHttpRequest'
}


def calculate_crop():
    """ crop product calculate request with potato product """
    with open(path.join(FILE_PATH, SAMPLES_PATH, 'crop_product-calculate-input.json'), 'r') as f:
        data = f.read()

    r = requests.post(API_URL + 'crop_product/calculate/', data, headers=HEADERS)

    try:
        print json.dumps(json.loads(r.content), indent=4)
    except ValueError, e:
        print r.content
        raise

def calculate_dairy():
    """ dairy product calculate request """

    with open(path.join(FILE_PATH, SAMPLES_PATH, 'dairy_product-calculate-input.json'), 'r') as f:
        data = f.read()

    r = requests.post(API_URL + 'dairy_product/calculate/', data, headers=HEADERS)


    print "dairy product: {0.status_code}".format(r)
    try:
        print json.dumps(json.loads(r.content), indent=4)
    except ValueError, e:
        print r.content
        raise

def calculate_energy():
    """ energy density calculate request """

    with open(path.join(FILE_PATH, SAMPLES_PATH, 'energy-calculate-input.json'), 'r') as f:
        data = json.loads(f.read())

    r = requests.post(API_URL + 'energy/calculate/', data, headers=HEADERS)

    print "energy: {0.status_code}".format(r)
    try:
        print json.dumps(json.loads(r.content), indent=4)
    except ValueError, e:
        print r.content
        raise

def calculate_fertiliser():
    """ fertiliser nutrient calculate request """

    with open(path.join(FILE_PATH, SAMPLES_PATH, 'fertiliser-calculate-input.json'), 'r') as f:
        data = json.loads(f.read())

    r = requests.post(API_URL + 'fertiliser/calculate/', data, headers=HEADERS)

    print "fertiliser: {0.status_code}".format(r)
    try:
        print json.dumps(json.loads(r.content), indent=4)
    except ValueError, e:
        print r.content
        raise

def calculate_irrigation():
    """ irrigation water usage calculation request """

    with open(path.join(FILE_PATH, SAMPLES_PATH, 'irrigation-calculate-input.json'), 'r') as f:
        data = json.loads(f.read())

    r = requests.post(API_URL + 'irrigation/calculate/', data, headers=HEADERS)

    print "irrigation: {0.status_code}".format(r)
    try:
        print json.dumps(json.loads(r.content), indent=4)
    except ValueError, e:
        print r.content
        raise


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--crop', action='store_true',
                    help='calculate crops')
parser.add_argument('--dairy', action='store_true',
                    help='calculate dairy')
parser.add_argument('--energy', action='store_true',
                    help='calculate energy use')
parser.add_argument('--irrigation', action='store_true',
                    help='calculate irrigation water use')
parser.add_argument('--fertiliser', action='store_true',
                    help='calculate fertiliser nutrients')
parser.add_argument('--all', action='store_true',
                    help='calculate all results')

args = parser.parse_args()

for k in ['crop', 'dairy', 'fertiliser', 'irrigation']:
    if args.all or getattr(args, k, False):
        eval('calculate_{}'.format(k))()
