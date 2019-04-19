#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Please first complete the following steps:
#
#   1. Download
#      https://github.com/PCMDI/cmip6-cmor-tables.git
#      Create a soft link cmip6-cmor-tables/Tables to ./Tables in your
#      working directory
#
#   python CMIP6Validtor ../Tables/CMIP6_Amon.json ../CMIP6/yourfile.nc
#

"""
Created on Fri Feb 19 11:33:52 2016

@author: Denis Nadeau LLNL
@co-author: Guillaume Levavasseur (IPSL) Parallelization

"""

from __future__ import print_function

import argparse
import itertools
import json
import os
import re
import sys
from argparse import ArgumentTypeError
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool
from uuid import uuid4 as uuid
from collections import namedtuple
import cdms2
import cmip6_cv
import numpy
from fuzzywuzzy.fuzz import partial_ratio
from fuzzywuzzy.process import extractOne


# =========================
# BColors()
# =========================


class BColors:
    """
    Background colors for print statements

    """

    def __init__(self):
        self.HEADER = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKGREEN = '\033[1;32m'
        self.WARNING = '\033[1;34;47m'
        self.FAIL = '\033[1;31;47m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'


BCOLORS = BColors()


# =========================
# DIRAction()
# =========================
class DIRECTORYAction(argparse.Action):
    """
    Check if argparse is a directory.

    """

    def __call__(self, parser, namespace, values, option_string=None):
        prospective = values
        if not os.path.isdir(prospective):
            if not os.path.isfile(prospective):
                raise argparse.ArgumentTypeError(
                    'No such file/directory: {}'.format(prospective))

        if os.access(prospective, os.R_OK):
            setattr(namespace, self.dest, prospective)
        else:
            raise argparse.ArgumentTypeError(
                'Read access denied: {}'.format(prospective))


# =========================
# INPUTAction()
# =========================
class INPUTAction(argparse.Action):
    """
    Checks if the supplied input exists.

    """

    def __call__(self, parser, namespace, values, option_string=None):
        checked_values = [self.input_checker(x) for x in values]
        setattr(namespace, self.dest, checked_values)

    @staticmethod
    def input_checker(path):
        path = os.path.abspath(os.path.normpath(path))
        if not os.path.exists(path):
            msg = 'No such input: {}'.format(path)
            raise argparse.ArgumentTypeError(msg)
        return path


# =========================
# Collector()
# =========================
class Collector(object):
    """
    Base collector class to yield regular NetCDF files.

    :param list sources: The list of sources to parse
    :returns: The data collector
    :rtype: *iter*

    """

    def __init__(self, sources):
        self.sources = sources
        self.FileFilter = FilterCollection()
        self.PathFilter = FilterCollection()
        assert isinstance(self.sources, list)

    def __iter__(self):
        for source in self.sources:
            if os.path.isdir(source):
                # If input is a directory: walk through it and yields netCDF files
                for root, _, filenames in os.walk(source, followlinks=True):
                    if self.PathFilter(root):
                        for filename in sorted(filenames):
                            ffp = os.path.join(root, filename)
                            if os.path.isfile(ffp) and self.FileFilter(filename):
                                yield ffp
            else:
                # It input is a file: yields the netCDF file itself
                root, filename = os.path.split(source)
                if self.PathFilter(root) and self.FileFilter(filename):
                    yield source

    def __len__(self):
        """
        Returns collector length.

        :returns: The number of items in the collector.
        :rtype: *int*

        """
        s = 0
        for _ in self.__iter__():
            s += 1
        return s


class FilterCollection(object):
    """
    Regex dictionary with a call method to evaluate a string against several regular expressions.
    The dictionary values are 2-tuples with the regular expression as a string and a boolean
    indicating to match (i.e., include) or non-match (i.e., exclude) the corresponding expression.

    """
    if sys.version_info < (3, 0):
        FILTER_TYPES = (str, re._pattern_type)
    else:
        FILTER_TYPES = (str, re.Pattern)

    def __init__(self):
        self.filters = dict()

    def add(self, name=None, regex='*', inclusive=True):
        """Add new filter"""
        if not name:
            name = uuid()
        assert isinstance(regex, self.FILTER_TYPES)
        assert isinstance(inclusive, bool)
        self.filters[name] = (regex, inclusive)

    def __call__(self, string):
        return all([self.match(regex, string, inclusive=inclusive) for regex, inclusive in list(self.filters.values())])

    @staticmethod
    def match(pattern, string, inclusive=True):
        # Assert inclusive and exclusive flag are mutually exclusive
        if inclusive:
            return True if re.search(pattern, string) else False
        else:
            return True if not re.search(pattern, string) else False


# =========================
# Print Error method()
# =========================
def print_err(msg):
    print(BCOLORS.FAIL)
    print("==========================================================================")
    print(msg)
    print("==========================================================================")
    print(BCOLORS.ENDC)


# =========================
# checkCMIP6()
# =========================
class checkCMIP6(object):
    """
    Validate if a file is CMIP6 compliant and ready for publication.

    Class need to read CMIP6 Table and Controled Vocabulary file.

    As well,the class will load the EXPERIMENT json file

    Input:
        args.cmip6_table:  CMIP6 table used to creat this file,
                           variable attributes and dimensions will be controled.
        args.CV:           Controlled Vocabulary "json" file.

    Output:
        outfile:      Log file, default is stdout.

    """

    def __init__(self, table_path):
        # -------------------------------------------------------------------
        #  Reset CV error switch
        # -------------------------------------------------------------------
        self.errors = 0
        # -------------------------------------------------------------------
        #  Initialize table path
        # -------------------------------------------------------------------
        self.cmip6_table_path = os.path.normpath(table_path)
        # -------------------------------------------------------------------
        #  Initialize attributes dictionaries
        # -------------------------------------------------------------------
        self.dictGbl = dict()
        self.dictVar = dict()
        self.dictAxis = dict()
        # -------------------------------------------------------------------
        # call setup() to clean all 'C' internal memory.
        # -------------------------------------------------------------------
        cmip6_cv.setup(inpath="../Tables", exit_control=cmip6_cv.CMOR_EXIT_ON_WARNING)
        # -------------------------------------------------------------------
        # Set Control Vocabulary file to use (default from cmor.h)
        # -------------------------------------------------------------------
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.GLOBAL_CV_FILENAME,
            cmip6_cv.TABLE_CONTROL_FILENAME)
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.FILE_PATH_TEMPLATE,
            cmip6_cv.CMOR_DEFAULT_PATH_TEMPLATE)
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.FILE_NAME_TEMPLATE,
            cmip6_cv.CMOR_DEFAULT_FILE_TEMPLATE)
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.GLOBAL_ATT_FURTHERINFOURLTMPL,
            cmip6_cv.CMOR_DEFAULT_FURTHERURL_TEMPLATE)
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.CMOR_AXIS_ENTRY_FILE,
            "CMIP6_coordinate.json")
        cmip6_cv.set_cur_dataset_attribute(
            cmip6_cv.CMOR_FORMULA_VAR_FILE,
            "CMIP6_formula_terms.json")
        # -------------------------------------------------------------------
        # Set axis attributes to ignore & skip
        # -------------------------------------------------------------------
        self._AXIS_IGNORED_ATTRS = ["long_name",
                                    "comment",
                                    "requested",
                                    "out_name",
                                    "must_have_bounds",
                                    "bounds_values",
                                    "value",
                                    "z_bounds_factors",
                                    "valid_min",
                                    "tolerance",
                                    "z_factors",
                                    "type",
                                    "requested_bounds",
                                    "stored_direction",
                                    "generic_level_name",
                                    "valid_max"]

    @staticmethod
    def _check_json_table(path):
        f = open(path)
        lines = f.readlines()
        f.close()
        jsonobject = json.loads(" ".join(lines))
        if not jsonobject:
            raise argparse.ArgumentTypeError(
                'Invalid JSON CMOR table: {}'.format(path))
        return jsonobject

    def set_double_value(self, attribute):
        if cmip6_cv.has_cur_dataset_attribute(attribute):
            if isinstance(self.dictGbl[attribute], numpy.ndarray) and \
                    isinstance(self.dictGbl[attribute][0], numpy.float64):
                self.dictGbl[attribute] = self.dictGbl[attribute][0]
                cmip6_cv.set_cur_dataset_attribute(attribute, self.dictGbl[attribute])

    @staticmethod
    def _test_var_is_climatology(infile, **kwargs):
        return 'Clim' if os.path.basename(infile.id).find('-clim') != -1 else None

    @staticmethod
    def _test_var_has_3_dimensions(infile, variable, **kwargs):
        return '2d' if len(infile.variables[variable].listdimnames()) == 3 else None

    @staticmethod
    def _test_var_has_27_pressure_levels(infile, **kwargs):
        return '27' if 'plev' in infile.listdimension() and int(infile.axes['plev'].shape[0]) == 27 else None

    @staticmethod
    def _test_var_has_7_pressure_levels(infile, **kwargs):
        return '7h' if 'plev' in infile.listdimension() and int(infile.axes['plev'].shape[0]) == 7 else None

    @staticmethod
    def _test_var_has_4_pressure_levels(infile, **kwargs):
        return '4' if 'plev' in infile.listdimension() and int(infile.axes['plev'].shape[0]) == 4 else None

    @staticmethod
    def _test_var_has_land_in_cell_methods(infile, variable, **kwargs):
        return 'land' if 'land' in infile.variables[variable].cell_methods else None

    @staticmethod
    def _test_axis_time(infile, **kwargs):
        if 'time' in infile.listdimension():
            time = infile.axes['time']
            if 'bounds' not in time.attributes:
                return 'time1'
            elif 'climatology' in time.attributes:
                return 'time2'
            else:
                return 'time'
        else:
            return None

    @staticmethod
    def _test_axis_height(infile, **kwargs):
        if 'height' in infile.listdimension() and int(infile.axes['height'].shape[0]) == 1:
            height = infile.axes['height']
            if height.getvalue() == 2.:
                return 'height2m'
            elif height.getvalue() == 10.:
                return 'height10m'
            elif height.getvalue() == 100.:
                return 'height100m'
            else:
                return None
        else:
            return None

    @staticmethod
    def _test_axis_line(infile, **kwargs):
        if 'line' in infile.listdimension():
            if int(infile.axes['line'].shape[0]) >= 6:
                return 'oline'
            else:
                return 'sline'
        else:
            return None

    @staticmethod
    def _test_axis_type(**kwargs):
        # No "type" coordinate reflects a vector of PFT of vegetation type in CMIP6_coordinate.json.
        # Set "type" equivalent to "vegtype" by default
        # Consequently other "type" out_names won't be checked in any case as "type" is always mapped to "vegtype".
        return 'vegtype'

    @staticmethod
    def _test_axis_plev(infile, **kwargs):
        if 'plev' in infile.listdimension():
            plev = infile.axes['plev']
            if int(plev.shape[0]) == 1:
                if plev.getValue() == 1000.:
                    return 'p10'
                elif plev.getValue() == 10000.:
                    return 'p100'
                elif plev.getValue() == 100000.:
                    return 'p1000'
                elif plev.getValue() == 20000.:
                    return 'p200'
                elif plev.getValue() == 22000.:
                    return 'p220'
                elif plev.getValue() == 50000.:
                    return 'p500'
                elif plev.getValue() == 56000.:
                    return 'p560'
                elif plev.getValue() == 70000.:
                    if 'bounds' in plev.attributes:
                        return 'pl700'
                    else:
                        return 'p700'
                elif plev.getValue() == 84000.:
                    return 'p840'
                elif plev.getValue() == 85000.:
                    return 'p850'
                else:
                    return None
            elif int(plev.shape[0]) == 3:
                if 'bounds' in plev.attributes:
                    return 'plev3'
                else:
                    return 'plev3h'
            elif int(plev.shape[0]) == 4:
                return 'plev4'
            elif int(plev.shape[0]) == 7:
                if 'bounds' in plev.attributes:
                    return 'plev7'
                else:
                    return 'plev7h'
            elif int(plev.shape[0]) == 8:
                return 'plev8'
            elif int(plev.shape[0]) == 19:
                return 'plev19'
            elif int(plev.shape[0]) == 23:
                return 'plev23'
            elif int(plev.shape[0]) == 27:
                return 'plev27'
            elif int(plev.shape[0]) == 39:
                return 'plev39'
            else:
                return None
        else:
            return None

    @staticmethod
    def _test_axis_depth(infile, **kwargs):
        if 'depth' in infile.listdimension() and int(infile.axes['depth'].shape[0]) == 1:
            depth = infile.axes['depth']
            if depth.getValue() == 0.:
                return 'depth0m'
            elif depth.getValue() == 100.:
                return 'depth100m'
            elif depth.getValue() == 1000.:
                return 'depth2000m'
            elif depth.getValue() == 150.:
                return 'depth300m'
            elif depth.getValue() == 350.:
                return 'depth700m'
            elif depth.getValue() == 50.:
                return 'olayer100m'
            elif depth.getValue() == '':
                return 'sdepth'
            elif depth.getValue() == 0.05:
                return 'sdepth1'
            elif depth.getValue() == 0.5:
                return 'sdepth10'
            else:
                return None
        else:
            return None

    @staticmethod
    def _test_axis_lev(infile, names, **kwargs):
        if 'lev' in infile.listdimension():
            if 'long_name' not in infile.axes['lev'].attributes:
                return None
            long_name = infile.axes['lev'].long_name
            long_names = zip(*names)[1]
            key, score = extractOne(long_name, long_names, scorer=partial_ratio)
            if score >= 80:
                idx = long_names.index(long_name)
                if 'bounds' in infile.axes['lev'].attributes:
                    return '{}_half'.format(names[idx][0])
                else:
                    return names[idx][0]
            else:
                return None
        else:
            return None

    def _find_cmor_entry(self, variable, infile, cmor_dict, axis=False):
        # -------------------------------------------------------------------
        #  Distinguish similar CMOR entries with the same out_name if exist
        # -------------------------------------------------------------------
        # The goal is to deduce here the CMOR entry to consider for the check.
        # Some variables (called "out_names" into CMOR table) corresponds to several CMOR entries in the same table.
        # i.e., several CMOR entries have the same out_name value.
        # 1. For the considered variable --> get all (CMOR entries, out_name) pairs in the CMOR table
        #    where out_name = variable
        var_entries = [(f, cmor_dict[f]['out_name']) for f in cmor_dict if cmor_dict[f]['out_name'] == variable]
        # 2. Raise an error if no corresponding out_name found (i.e., var_entires = [])
        if not var_entries:
            print_err("CMOR entry with out_name {} could not be found in CMOR table".format(variable))
            raise KeyboardInterrupt
        # 2. If only one pair --> consider this entry
        if len(var_entries) == 1:
            cmor_entry = var_entries[0][0]
        else:
            # 3. If several entries --> apply additional tests to the file
            #    (i.e., is_climatology, has_7_pressure_levels, etc.)
            # Make "variable" as CMOR entry to consider by default/rollback in case of no successful test.
            cmor_entry = variable
            if not axis:
                for test in [t for t in dir(self) if t.startswith('_test_var')]:
                    suffix = getattr(self, test)(**{'infile': infile, 'variable': variable})
                    # If one test is successful, get the appropriate suffix
                    # Test if "variable" + "suffix" exists as a CMOR entry
                    if suffix and variable + suffix in cmor_dict:
                        cmor_entry = variable + suffix
                        # Break the loop as no variable passes several tests.
                        break
            else:
                # Extract long_name attribute to help cmor_entry discrimination.
                long_names = list()
                if variable == 'lev':
                    long_names = [(f, cmor_dict[f]['long_name']) for f in cmor_dict if
                                  cmor_dict[f]['out_name'] == variable]
                # Apply test related to the axis name.
                if hasattr(self, '_test_axis_{}'):
                    entry = getattr(self, '_test_axis_{}'.format(variable))(**{'infile': infile,
                                                                               'variable': variable,
                                                                               'names': long_names})
                    # Return entry if not None
                    if entry:
                        cmor_entry = entry
        return cmor_entry

    def ControlVocab(self, ncfile, variable=None, print_all=True):
        """
        Check CMIP6 global attributes against Control Vocabulary file.

            1. Validate required attribute if presents and some values.
            2. Validate registered institution and institution_id
            3. Validate registered source and source_id
            4. Validate experiment, experiment_id and all attributes associated with this experiment.
                   Make sure that all attributes associate with the experiment_id found in CMIP6_CV.json
                   are set to the appropriate values.
            5. Validate grid_label and grid_resolution
            6. Validate creation time in ISO format (YYYY-MM-DDTHH:MM:SS)
            7. Validate furtherinfourl from CV internal template
            8. Validate variable attributes with CMOR JSON table.
            9. Validate parent_* attribute
           10. Validate sub_experiment_* attributes.
           11. Validate that all *_index are integers.

        """
        # -------------------------------------------------------------------
        #  Open file in processing
        # -------------------------------------------------------------------
        infile = cdms2.open(ncfile, "r")
        # Check that data_specs_version, table_id and variable_id attributes exist.
        if not infile.getglobal('variable_id'):
            print_err("The global attribute 'variable_id' could not be found in file.")
            raise KeyboardInterrupt
        if not infile.getglobal('table_id'):
            print_err("The global attribute 'table_id' could not be found in file.")
            raise KeyboardInterrupt
        if not infile.getglobal('data_specs_version'):
            print_err("The global attribute 'data_specs_version' could not be found in file.")
            raise KeyboardInterrupt
        # -------------------------------------------------------------------
        #  Initialize arrays
        # -------------------------------------------------------------------
        # If table_path is the table directory
        # Deduce corresponding JSON MIP table
        if os.path.isdir(self.cmip6_table_path):
            # In case automated DR version switch is implemented elsewhere out of PrePARE
            # Skip if the --table-path ends with "Tables" or a DR tag (e.g., 01.00.28)
            if self.cmip6_table_path.endswith('Tables') or self.cmip6_table_path.endswith(infile.data_specs_version):
                cmip6_table = '{}/CMIP6_{}.json'.format(
                    self.cmip6_table_path,
                    infile.table_id)
                cmip6_coordinates = '{}/CMIP6_coordinate.json'.format(
                    self.cmip6_table_path)
            else:
                cmip6_table = '{}/{}/CMIP6_{}.json'.format(
                    self.cmip6_table_path,
                    infile.data_specs_version,
                    infile.table_id)
                cmip6_coordinates = '{}/{}/CMIP6_coordinate.json'.format(
                    self.cmip6_table_path,
                    infile.data_specs_version)
        else:
            cmip6_table = self.cmip6_table_path
            cmip6_coordinates = os.path.join(os.path.dirname(self.cmip6_table_path), 'CMIP6_coordinate.json')
        table_id = os.path.basename(os.path.splitext(cmip6_table)[0]).split('_')[1]
        # Check and get JSON table
        cmor_table = self._check_json_table(cmip6_table)
        # Check and get coordinates table
        coordinates_table = self._check_json_table(cmip6_coordinates)
        # -------------------------------------------------------------------
        # Load CMIP6 table into memory
        # -------------------------------------------------------------------
        table = cmip6_cv.load_table(cmip6_table)
        # -------------------------------------------------------------------
        #  Deduce variable
        # -------------------------------------------------------------------
        # If variable can be deduced from the file attributes (Default)
        # If not variable submitted on command line with --variable is considered
        variable_id = infile.variable_id
        if not variable:
            variable = variable_id
        # -------------------------------------------------------------------
        #  Get CMOR entry corresponding to the file variable
        # -------------------------------------------------------------------
        cmor_entries = cmor_table['variable_entry']
        variable_cmor_entry = self._find_cmor_entry(variable=variable,
                                                    infile=infile,
                                                    cmor_dict=cmor_entries)
        #  -------------------------------------------------------------------
        #  Get variable out name in netCDF record
        #  -------------------------------------------------------------------
        # Variable record name should follow CMOR table out names
        if variable_cmor_entry not in list(cmor_entries.keys()):
            print_err("The entry " + variable_cmor_entry + " could not be found in CMOR table")
            raise KeyboardInterrupt
        variable_record_name = cmor_entries[variable_cmor_entry]['out_name']
        # Variable id attribute should be the same as variable record name
        # in any case to be CF- and CMIP6-compliant
        variable_id = variable_record_name
        # -------------------------------------------------------------------
        # Create a dictionary of all global attributes
        # -------------------------------------------------------------------
        self.dictGbl = infile.attributes
        for key, value in list(self.dictGbl.items()):
            cmip6_cv.set_cur_dataset_attribute(key, value)
        # Set member_id attribute depending on sub_experiment_id and variant_label
        member_id = ""
        if "sub_experiment_id" in list(self.dictGbl.keys()):
            if self.dictGbl["sub_experiment_id"] not in ['none']:
                member_id = '{}-{}'.format(self.dictGbl['sub_experiment_id'],
                                           self.dictGbl['variant_label'])
            else:
                member_id = self.dictGbl['variant_label']
        cmip6_cv.set_cur_dataset_attribute(cmip6_cv.GLOBAL_ATT_MEMBER_ID, member_id)
        # -------------------------------------------------------------------
        # Create a dictionary of attributes for the variable
        # -------------------------------------------------------------------
        try:
            self.dictVar = infile.variables[variable_record_name].attributes
        except BaseException:
            print_err("The variable " + variable_record_name + " could not be found in file")
            raise KeyboardInterrupt

        # -------------------------------------------------------------------
        # Check global attributes
        # -------------------------------------------------------------------
        self.errors += cmip6_cv.check_requiredattributes(table)
        self.errors += cmip6_cv.check_institution(table)
        self.errors += cmip6_cv.check_sourceID(table)
        self.errors += cmip6_cv.check_experiment(table)
        self.errors += cmip6_cv.check_grids(table)
        self.errors += cmip6_cv.check_ISOTime()
        self.errors += cmip6_cv.check_furtherinfourl(table)
        self.errors += cmip6_cv.check_subExpID(table)
        for attr in ['branch_time_in_child', 'branch_time_in_parent']:
            if attr in list(self.dictGbl.keys()):
                self.set_double_value(attr)
                if not isinstance(self.dictGbl[attr], numpy.float64):
                    print_err("{} is not a double: {}".format(attr, type(self.dictGbl[attr])))
                    self.errors += 1
        for attr in ['realization_index', 'initialization_index', 'physics_index', 'forcing_index']:
            if not isinstance(self.dictGbl[attr], numpy.ndarray):
                print_err("{} is not an integer: {}".format(attr, type(self.dictGbl[attr])))
                self.errors += 1
        self.errors += cmip6_cv.check_parentExpID(table)
        for attr in ['table_id', 'variable_id']:
            try:
                if locals()[attr] != self.dictGbl[attr]:
                    print_err("{} attribute is not consistent: {}".format(attr, self.dictGbl[attr]))
                    self.errors += 1
            except KeyError:
                print_err("{} attribute is missing in global attributes".format(attr))
                self.errors += 1

        # -------------------------------------------------------------------
        # Check variable coordinates
        # -------------------------------------------------------------------
        cmor_entries = coordinates_table['axis_entry']
        # Get variable shape from infile
        axis_ids = infile.variables[variable_record_name].getAxisIds()
        # Check variable coordinates
        for axis_id in axis_ids:
            # -------------------------------------------------------------------
            #  Get CMOR entry corresponding to the axis id
            # -------------------------------------------------------------------
            axis_cmor_entry = self._find_cmor_entry(variable=axis_id,
                                                    infile=infile,
                                                    cmor_dict=cmor_entries,
                                                    axis=True)
            #  -------------------------------------------------------------------
            #  Get axis out name in netCDF record
            #  -------------------------------------------------------------------
            # Axis record name should follow CMOR table out names
            if axis_cmor_entry not in list(cmor_entries.keys()):
                print_err("The entry {} could not be found in CMOR table".format(axis_cmor_entry))
                raise KeyboardInterrupt
            cv_attrs = cmor_entries[axis_cmor_entry]
            # Axis id attribute should be the same as axis record name
            # in any case to be CF- and CMIP6-compliant
            axis_id = cv_attrs['out_name']
            #  -------------------------------------------------------------------
            #  Get axis CV attr
            #  -------------------------------------------------------------------
            self.dictAxis = infile.axes[axis_id].attributes
            #  -------------------------------------------------------------------
            #  Check axis CV attr
            #  -------------------------------------------------------------------
            for key in cv_attrs:
                if key in self._AXIS_IGNORED_ATTRS:
                    continue
                # Is this attribute in file?
                if key in list(self.dictAxis.keys()):
                    # Verify that attribute value is equal to file attribute
                    table_value = cv_attrs[key]
                    file_value = self.dictAxis[key]
                    # PrePARE accept units of 1 or 1.0 so adjust the table_value
                    if key == "units":
                        if "time" in axis_id and infile.axes[axis_id].units.startswith("days since"):
                            table_value = file_value
                        if (table_value == "1") and (file_value == "1.0"):
                            table_value = "1.0"
                        if (table_value == "1.0") and (file_value == "1"):
                            table_value = "1"
                    if isinstance(table_value, str) and isinstance(file_value, numpy.ndarray):
                        if numpy.array([int(value) for value in table_value.split()] == file_value).all():
                            file_value = True
                            table_value = True
                    if isinstance(table_value, numpy.ndarray):
                        table_value = table_value[0]
                    if isinstance(file_value, numpy.ndarray):
                        file_value = file_value[0]
                    if isinstance(table_value, float):
                        if abs(table_value - file_value) <= 0.00001 * abs(table_value):
                            table_value = file_value
                    if key == "climatology" and self._test_var_is_climatology(infile):
                        table_value = "climatology_bounds"
                    if str(table_value) != str(file_value):
                        print_err("Your file contains \"{0}: \"{1}\" and\n "
                                  "CMIP6 tables requires \"{0}\": \"{2}\".".format(key,
                                                                                   str(file_value),
                                                                                   str(table_value)))
                        self.errors += 1
                elif cv_attrs[key] != "":
                    # That attribute is not in the axis metadata
                    table_value = cv_attrs[key]
                    if isinstance(table_value, numpy.ndarray):
                        table_value = table_value[0]
                    if isinstance(table_value, float):
                        table_value = "{0:.2g}".format(table_value)
                    print_err("CMIP6 axis {} requires \"{}\": \"{}\".".format(axis_id, key, str(table_value)))
                    self.errors += 1

        # -------------------------------------------------------------------
        # Get time axis properties
        # -------------------------------------------------------------------
        # Get calendar and time units
        try:
            calendar = infile.axes['time'].calendar
            timeunits = infile.axes['time'].units
        except BaseException:
            calendar = "gregorian"
            timeunits = "days since ?"
        # Get first and last time bounds
        climatology = self._test_var_is_climatology(infile)
        if climatology:
            if cmip6_table.find('Amon') != -1:
                variable = '{}Clim'.format(variable)

        clim_idx = variable.find('Clim')
        if climatology and clim_idx != -1:
            var = [variable[:clim_idx]]

        try:
            if 'bounds' in list(infile.axes['time'].attributes.key()):
                bndsvar = infile.axes['time'].bounds
            elif 'climatology' in list(infile.axes['time'].attributes.keys()):
                bndsvar = infile.axes['time'].climatology
            else:
                bndsvar = 'time_bnds'
            startimebnds = infile.variables[bndsvar][0][0]
            endtimebnds = infile.variables[bndsvar][-1][1]
        except BaseException:
            startimebnds = 0
            endtimebnds = 0

        try:
            startime = infile.axes['time'][0]
            endtime = infile.axes['time'][-1]
        except BaseException:
            startime = 0
            endtime = 0

        # -------------------------------------------------------------------
        # Setup variable
        # -------------------------------------------------------------------
        varid = cmip6_cv.setup_variable(variable_cmor_entry,
                                        self.dictVar['units'],
                                        self.dictVar['_FillValue'][0],
                                        startime,
                                        endtime,
                                        startimebnds,
                                        endtimebnds)
        if varid == -1:
            print_err("Could not find variable {} in table {} ".format(variable_cmor_entry, cmip6_table))
            raise KeyboardInterrupt
        # -------------------------------------------------------------------
        # Check filename
        # -------------------------------------------------------------------
        self.errors += cmip6_cv.check_filename(table,
                                               varid,
                                               calendar,
                                               timeunits,
                                               os.path.basename(infile.id))
        # -------------------------------------------------------------------
        # Check variable attributes
        # -------------------------------------------------------------------
        cv_attrs = cmip6_cv.list_variable_attributes(varid)
        for key in cv_attrs:
            if key == "long_name":
                continue
            if key == "comment":
                continue
            if key == "cell_measures":
                if cv_attrs[key].find("OPT") != -1 or cv_attrs[key].find("MODEL") != -1:
                    continue
            # Is this attribute in file?
            if key in list(self.dictVar.keys()):
                # Verify that attribute value is equal to file attribute
                table_value = cv_attrs[key]
                file_value = self.dictVar[key]
                # PrePARE accept units of 1 or 1.0 so adjust the table_value
                if key == "units":
                    if (table_value == "1") and (file_value == "1.0"):
                        table_value = "1.0"
                    if (table_value == "1.0") and (file_value == "1"):
                        table_value = "1"
                if isinstance(table_value, str) and isinstance(file_value, numpy.ndarray):
                    if numpy.array([int(value) for value in table_value.split()] == file_value).all():
                        file_value = True
                        table_value = True
                if isinstance(table_value, numpy.ndarray):
                    table_value = table_value[0]
                if isinstance(file_value, numpy.ndarray):
                    file_value = file_value[0]
                if isinstance(table_value, float):
                    if abs(table_value - file_value) <= 0.00001 * abs(table_value):
                        table_value = file_value
                if key == "cell_methods":
                    idx = file_value.find(" (")
                    if idx != -1:
                        file_value = file_value[:idx]
                        table_value = table_value[:idx]
                if key == "cell_measures":
                    pattern = re.compile('(?P<param>[\w.-]+): (?P<val1>[\w.-]+) OR (?P<val2>[\w.-]+)')
                    values = re.findall(pattern, table_value)
                    table_values = [""]  # Empty string is allowed in case of useless attribute
                    if values:
                        tmp = dict()
                        for param, val1, val2 in values:
                            tmp[param] = [str('{}: {}'.format(param, val1)), str('{}: {}'.format(param, val2))]
                        table_values.extend([' '.join(i) for i in list(itertools.product(*list(tmp.values())))])
                        if str(file_value) not in list(map(str, table_values)):
                            print_err("Your file contains \"{0}: \"{1}\" and\n "
                                      "CMIP6 tables requires \"{0}\": \"{2}\".".format(key,
                                                                                       str(file_value),
                                                                                       str(table_value)))
                            self.errors += 1
                        continue

                if str(table_value) != str(file_value):
                    print_err("Your file contains \"{0}: \"{1}\" and\n "
                              "CMIP6 tables requires \"{0}\": \"{2}\".".format(key,
                                                                               str(file_value),
                                                                               str(table_value)))
                    print(BCOLORS.ENDC)
                    self.errors += 1
            else:
                # That attribute is not in the file
                table_value = cv_attrs[key]
                if isinstance(table_value, numpy.ndarray):
                    table_value = table_value[0]
                if isinstance(table_value, float):
                    table_value = "{0:.2g}".format(table_value)
                print_err(
                    "CMIP6 variable {} requires \"{}\": \"{}\".".format(variable_cmor_entry, key, str(table_value)))
                self.errors += 1
        # Print final message
        if self.errors != 0:
            raise KeyboardInterrupt
        elif print_all:
            print(BCOLORS.OKGREEN + "     :: CV SUCCESS :: {}".format(ncfile) + BCOLORS.ENDC)


def process(source):
    # Redirect all print statements to a logfile dedicated to the current
    # process
    logfile = '/tmp/PrePARE-{}.log'.format(os.getpid())
    with RedirectedOutput(logfile):
        rc = sequential_process(source)
    # Close and return logfile
    return logfile, rc


def sequential_process(source):
    # Get context from global process env
    assert 'pctx' in list(globals().keys())
    pctx = globals()['pctx']
    try:
        # Process file
        checker = checkCMIP6(pctx.table_path)
        if pctx.variable:
            checker.ControlVocab(source, variable=pctx.variable, print_all=pctx.all)
        else:
            checker.ControlVocab(source, print_all=pctx.all)
        return 0
    except KeyboardInterrupt:
        print(BCOLORS.FAIL + "└──> :: CV FAIL    :: {}".format(source) + BCOLORS.ENDC)
        return 1
    except Exception as e:
        print(e)
        msg = BCOLORS.WARNING
        msg += "└──> :: SKIPPED    :: {}".format(source)
        msg += BCOLORS.ENDC
        print(msg)
        return 1
    finally:
        # Close opened file
        if hasattr(checker, "infile"):
            checker.infile.close()


def initializer(keys, values):
    """
    Initialize process context by setting particular variables as global variables.

    :param list keys: Argument name list
    :param list values: Argument value list

    """
    assert len(keys) == len(values)
    global pctx
    pctx = namedtuple("ChildProcess", keys)(*values)


def regex_validator(string):
    """
    Validates a Python regular expression syntax.

    :param str string: The string to check
    :returns: The Python regex
    :rtype: *re.compile*
    :raises Error: If invalid regular expression

    """
    try:
        return re.compile(string)
    except re.error:
        msg = 'Bad regex syntax: {}'.format(string)
        raise ArgumentTypeError(msg)


def processes_validator(value):
    """
    Validates the max processes number.

    :param str value: The max processes number submitted
    :return:
    """
    pnum = int(value)
    if pnum < 1 and pnum != -1:
        msg = 'Invalid processes number. Should be a positive integer or "-1".'
        raise ArgumentTypeError(msg)
    if pnum == -1:
        # Max processes = None corresponds to cpu.count() in Pool creation
        return None
    else:
        return pnum


#  =========================
#   main()
#  =========================
def main():
    parser = argparse.ArgumentParser(
        prog='PrePARE',
        description='Validate CMIP6 file for ESGF publication.')

    parser.add_argument(
        '-l', '--log',
        metavar='CWD',
        type=str,
        const='{}/logs'.format(os.getcwd()),
        nargs='?',
        help='Logfile directory. Default is the working directory.\n'
             'If not, standard output is used. Only available in multiprocessing mode.')

    parser.add_argument(
        '--variable',
        help='Specify geophysical variable name.\n'
             'If not variable is deduced from filename.')

    parser.add_argument(
        '--table-path',
        action=DIRECTORYAction,
        default=os.environ['CMIP6_CMOR_TABLES'] if 'CMIP6_CMOR_TABLES' in list(os.environ.keys()) else './Tables',
        help='Specify the CMIP6 CMOR tables path (JSON file).\n'
             'If not submitted read the CMIP6_CMOR_TABLES environment variable if exists.\n'
             'If a directory is submitted table is deduced from filename (default is "./Tables").\n'
             'If the submitted directory do not end with "Tables", the table path expected is \n'
             '"<table-dir>/<data_specs_version>" where "data_specs_version" is deduced from global \n'
             'attributes of each processed file.')

    parser.add_argument(
        '--max-processes',
        metavar='4',
        type=processes_validator,
        default=4,
        help='Number of maximal processes to simultaneously treat several files.\n'
             'Set to one seems sequential processing (default). Set to "-1" seems\n'
             'all available resources as returned by "multiprocessing.cpu_count()".')

    parser.add_argument(
        '--all',
        action='store_true',
        default=False,
        help='Show all results. Default only shows error(s) (i.e., file(s) not compliant)')

    parser.add_argument(
        '--ignore-dir',
        metavar="PYTHON_REGEX",
        type=str,
        default='^.*/\.[\w]*$',
        help='Filter directories NON-matching the regular expression.\n'
             'Default ignores paths with folder name(s) starting with "."')

    parser.add_argument(
        '--include-file',
        metavar='PYTHON_REGEX',
        type=regex_validator,
        action='append',
        help='Filter files matching the regular expression.\n'
             'Duplicate the flag to set several filters.\n'
             'Default only include NetCDF files.')

    parser.add_argument(
        '--exclude-file',
        metavar='PYTHON_REGEX',
        type=regex_validator,
        action='append',
        help='Filter files NON-matching the regular expression.\n'
             'Duplicate the flag to set several filters.\n'
             'Default only exclude hidden files (with names not\n'
             'starting with ".").')

    parser.add_argument(
        'input',
        nargs='+',
        action=INPUTAction,
        help='Input CMIP6 netCDF data to validate (ex: clisccp_cfMon_DcppC22_NICAM_gn_200001-200001.nc).\n'
             'If a directory is submitted all netCDF recursively found will be validate independently.')

    # Check command-line error
    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as errmsg:
        print(str(errmsg), file=sys.stderr)
        return 1
    except SystemExit:
        return 1
    # Get log
    logname = 'PrePARE-{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    log = None
    if args.log:
        if not os.path.isdir(args.log):
            os.makedirs(args.log)
        log = os.path.join(args.log, logname)
    # Collects netCDF files for process
    sources = Collector(args.input)
    # Set scan filters
    file_filters = list()
    if args.include_file:
        file_filters.extend([(f, True) for f in args.include_file])
    else:
        # Default includes netCDF only
        file_filters.append(('^.*\.nc$', True))
    if args.exclude_file:
        # Default exclude hidden files
        file_filters.extend([(f, False) for f in args.exclude_file])
    else:
        file_filters.append(('^\..*$', False))
    # Init collector file filter
    for regex, inclusive in file_filters:
        sources.FileFilter.add(regex=regex, inclusive=inclusive)
    # Init collector dir filter
    sources.PathFilter.add(regex=args.ignore_dir, inclusive=False)
    nb_sources = len(sources)
    errors = 0
    # Init process context
    cctx = dict()
    cctx['table_path'] = args.table_path
    cctx['variable'] = args.variable
    cctx['all'] = args.all
    # Separate sequential process and multiprocessing
    if args.max_processes != 1:
        # Create pool of processes
        pool = Pool(processes=args.max_processes, initializer=initializer,
                    initargs=(list(cctx.keys()), list(cctx.values())))
        # Run processes
        logfiles = list()
        progress = 0
        for logfile, rc in pool.imap(process, sources):
            progress += 1
            percentage = int(progress * 100 / nb_sources)
            msg = BCOLORS.OKGREEN + '\rCheck netCDF file(s): ' + BCOLORS.ENDC
            msg += '{}% | {}/{} files'.format(percentage, progress, nb_sources)
            sys.stdout.write(msg)
            sys.stdout.flush()
            logfiles.append(logfile)
            errors += rc
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
        # Print results from logfiles and remove them
        for logfile in set(logfiles):
            if not os.stat(logfile).st_size == 0:
                with open(logfile, 'r') as f:
                    if log:
                        with open(log, 'a+') as r:
                            r.write(f.read())
                    else:
                        sys.stdout.write(f.read())
                        sys.stdout.flush()
            os.remove(logfile)
        # Close pool of processes
        pool.close()
        pool.join()
    else:
        print('Checking data, please wait...')
        initializer(list(cctx.keys()), list(cctx.values()))
        for source in sources:
            errors += sequential_process(source)
    # Print results summary
    msg = BCOLORS.HEADER + '\nNumber of files scanned: {}'.format(nb_sources) + BCOLORS.ENDC
    if errors:
        msg += BCOLORS.FAIL
    else:
        msg += BCOLORS.OKGREEN
    msg += '\nNumber of file with error(s): {}'.format(errors) + BCOLORS.ENDC
    if log:
        with open(log, 'a+') as r:
            r.write(msg)
    print(msg)
    # Evaluate errors and exit with appropriate return code
    if errors != 0:
        if errors == nb_sources:
            # All files has error(s). Error code = -1
            sys.exit(-1)
        else:
            # Some files (at least one) has error(s). Error code = nb files with error(s)
            sys.exit(errors)
    else:
        # No errors. Error code = 0
        sys.exit(0)


@contextmanager
def RedirectedOutput(to=os.devnull):
    fd_out = sys.stdout.fileno()
    old_stdout = os.fdopen(os.dup(fd_out), 'w')
    fd_err = sys.stderr.fileno()
    old_stderr = os.fdopen(os.dup(fd_err), 'w')
    stream = open(to, 'a+')
    sys.stdout.close()
    sys.stderr.close()
    os.dup2(stream.fileno(), fd_out)
    os.dup2(stream.fileno(), fd_err)
    sys.stdout = os.fdopen(fd_out, 'w')
    sys.stderr = os.fdopen(fd_err, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        os.dup2(old_stdout.fileno(), fd_out)
        os.dup2(old_stderr.fileno(), fd_err)
        sys.stdout = os.fdopen(fd_out, 'w')
        sys.stderr = os.fdopen(fd_err, 'w')


if __name__ == '__main__':
    main()
