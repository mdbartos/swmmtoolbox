
# Example package with a console entry point
"""
Reads and formats data from the SWMM 5 output file.
"""
from __future__ import absolute_import
from __future__ import print_function

from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import sys
import struct
import datetime
import os
import math
import mmap

import mando
from mando.rst_text_formatter import RSTHelpFormatter
import pandas as pd

from tstoolbox import tsutils

PROPCODE = {0: {1: 'Area'},
            1: {0: 'Type',
                2: 'Inv_elev',
                3: 'Max_depth'},
            2: {0: 'Type',
                4: 'Inv_offset',
                3: 'Max_depth',
                5: 'Length'}}

# Names for the 'Node type' and 'Link type' codes above
TYPECODE = {0: {1: 'Area'},
            1: {0: 'Junction',  # nodes
                1: 'Outfall',
                2: 'Storage',
                3: 'Divider'},
            2: {0: 'Conduit',   # links
                1: 'Pump',
                2: 'Orifice',
                3: 'Weir',
                4: 'Outlet'}}

VARCODE = {0: {0: 'Rainfall',
               1: 'Snow_depth',
               2: 'Evaporation_loss',
               3: 'Infiltration_loss',
               4: 'Runoff_rate',
               5: 'Groundwater_outflow',
               6: 'Groundwater_elevation',
               7: 'Soil_moisture',
               8: 'Pollutant_washoff'},
           1: {0: 'Depth_above_invert',
               1: 'Hydraulic_head',
               2: 'Volume_stored_ponded',
               3: 'Lateral_inflow',
               4: 'Total_inflow',
               5: 'Flow_lost_flooding'},
           2: {0: 'Flow_rate',
               1: 'Flow_depth',
               2: 'Flow_velocity',
               3: 'Froude_number',
               4: 'Capacity'},
           4: {0: 'Air_temperature',
               1: 'Rainfall',
               2: 'Snow_depth',
               3: 'Evaporation_infiltration',
               4: 'Runoff',
               5: 'Dry_weather_inflow',
               6: 'Groundwater_inflow',
               7: 'RDII_inflow',
               8: 'User_direct_inflow',
               9: 'Total_lateral_inflow',
               10: 'Flow_lost_to_flooding',
               11: 'Flow_leaving_outfalls',
               12: 'Volume_stored_water',
               13: 'Evaporation_rate',
               14: 'Potential_PET'}}

# Prior to 5.10.10
VARCODE_OLD = {0: {0: 'Rainfall',
                   1: 'Snow_depth',
                   2: 'Evaporation_loss',
                   3: 'Runoff_rate',
                   4: 'Groundwater_outflow',
                   5: 'Groundwater_elevation'},
               1: {0: 'Depth_above_invert',
                   1: 'Hydraulic_head',
                   2: 'Volume_stored_ponded',
                   3: 'Lateral_inflow',
                   4: 'Total_inflow',
                   5: 'Flow_lost_flooding'},
               2: {0: 'Flow_rate',
                   1: 'Flow_depth',
                   2: 'Flow_velocity',
                   3: 'Froude_number',
                   4: 'Capacity'},
               4: {0: 'Air_temperature',
                   1: 'Rainfall',
                   2: 'Snow_depth',
                   3: 'Evaporation_infiltration',
                   4: 'Runoff',
                   5: 'Dry_weather_inflow',
                   6: 'Groundwater_inflow',
                   7: 'RDII_inflow',
                   8: 'User_direct_inflow',
                   9: 'Total_lateral_inflow',
                   10: 'Flow_lost_to_flooding',
                   11: 'Flow_leaving_outfalls',
                   12: 'Volume_stored_water',
                   13: 'Evaporation_rate'}}

# swmm_flowunits is here, but currently not used.
_SWMM_FLOWUNITS = {
    0: 'CFS',
    1: 'GPM',
    2: 'MGD',
    3: 'CMS',
    4: 'LPS',
    5: 'LPD'
}


_LOCAL_DOCSTRINGS = tsutils.docstrings
_LOCAL_DOCSTRINGS['filename'] = '''filename : str
        Filename of SWMM output file.  The SWMM model must complete
        successfully for "swmmtoolbox" to correctly read it.
        '''
_LOCAL_DOCSTRINGS['itemtype'] = '''itemtype : str
        One of 'system', 'node', 'link', or 'pollutant' to identify the
        type of data you want to extract.
        '''
_LOCAL_DOCSTRINGS['labels'] = '''labels : str
        The remaining arguments uniquely identify a time-series
        in the binary file.  The format is::

            'TYPE,NAME,VARINDEX'

        For example: 'node,C64,1 node,C63,1 ...'

        TYPE and NAME can be retrieved with::

            'swmmtoolbox list filename.out'

        VARINDEX can be retrieved with::

            'swmmtoolbox listvariables filename.out'
        '''


class SwmmExtract(object):
    """The class that handles all extraction of data from the out file."""
    def __init__(self, filename):

        self.RECORDSIZE = 4

        self.fp = open(filename, 'rb')

        self.fp.seek(-6 * self.RECORDSIZE, 2)

        self.Namesstartpos, \
            self.offset0, \
            self.startpos, \
            self.swmm_nperiods, \
            errcode, \
            magic2 = struct.unpack('6i', self.fp.read(6 * self.RECORDSIZE))

        self.fp.seek(0, 0)
        magic1 = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]

        if magic1 != 516114522:
            raise ValueError('''
*
*   First magic number incorrect.
*
''')
        if magic2 != 516114522:
            raise ValueError('''
*
*   Second magic number incorrect.
*
''')
        if errcode != 0:
            raise ValueError('''
*
*   Error code in output file indicates a problem with the run.
*
''')
        if self.swmm_nperiods == 0:
            raise ValueError('''
*
*   There are zero time periods in the output file.
*
''')

        # --- otherwise read additional parameters from start of file
        (version,
         self.swmm_flowunits,
         self.swmm_nsubcatch,
         self.swmm_nnodes,
         self.swmm_nlinks,
         self.swmm_npolluts) = struct.unpack('6i',
                                             self.fp.read(6 * self.RECORDSIZE))
        if version < 5100:
            self.varcode = VARCODE_OLD
        else:
            self.varcode = VARCODE

        self.itemlist = ['subcatchment', 'node', 'link', 'pollutant', 'system']

        # Read in the names
        self.fp.seek(self.Namesstartpos, 0)
        self.names = {0: [], 1: [], 2: [], 3: [], 4: []}
        number_list = [self.swmm_nsubcatch,
                       self.swmm_nnodes,
                       self.swmm_nlinks,
                       self.swmm_npolluts]
        for i, j in enumerate(number_list):
            for _ in range(j):
                stringsize = struct.unpack('i',
                                           self.fp.read(self.RECORDSIZE))[0]
                self.names[i].append(
                    struct.unpack('{0}s'.format(stringsize),
                                  self.fp.read(stringsize))[0])

        # Stupid Python 3
        for key in self.names:
            collect_names = []
            for name in self.names[key]:
                # Why would SMMM allow spaces in names?  Anyway...
                try:
                    rname = str(name, 'ascii', 'replace')
                except TypeError:
                    rname = name.decode('ascii', 'replace')
                try:
                    collect_names.append(rname.decode())
                except AttributeError:
                    collect_names.append(rname)
            self.names[key] = collect_names

        # Read pollutant concentration codes
        # = Number of pollutants * 4 byte integers
        self.pollutant_codes = struct.unpack(
            '{0}i'.format(self.swmm_npolluts),
            self.fp.read(self.swmm_npolluts * self.RECORDSIZE))

        self.propcode = {}
        self.prop = {0: [], 1: [], 2: []}
        nsubprop = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.propcode[0] = struct.unpack(
            '{0}i'.format(nsubprop),
            self.fp.read(nsubprop * self.RECORDSIZE))
        for i in range(self.swmm_nsubcatch):
            rprops = struct.unpack(
                '{0}f'.format(nsubprop),
                self.fp.read(nsubprop * self.RECORDSIZE))
            self.prop[0].append(list(zip(self.propcode[0], rprops)))

        nnodeprop = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.propcode[1] = struct.unpack(
            '{0}i'.format(nnodeprop),
            self.fp.read(nnodeprop * self.RECORDSIZE))
        for i in range(self.swmm_nnodes):
            rprops = struct.unpack(
                'i{0}f'.format(nnodeprop - 1),
                self.fp.read(nnodeprop * self.RECORDSIZE))
            self.prop[1].append(list(zip(self.propcode[1], rprops)))

        nlinkprop = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.propcode[2] = struct.unpack(
            '{0}i'.format(nlinkprop),
            self.fp.read(nlinkprop * self.RECORDSIZE))
        for i in range(self.swmm_nlinks):
            rprops = struct.unpack(
                'i{0}f'.format(nlinkprop - 1),
                self.fp.read(nlinkprop * self.RECORDSIZE))
            self.prop[2].append(list(zip(self.propcode[2], rprops)))

        self.vars = {}
        self.swmm_nsubcatchvars = struct.unpack(
            'i', self.fp.read(self.RECORDSIZE))[0]
        self.vars[0] = struct.unpack(
            '{0}i'.format(self.swmm_nsubcatchvars),
            self.fp.read(self.swmm_nsubcatchvars * self.RECORDSIZE))

        self.nnodevars = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.vars[1] = struct.unpack(
            '{0}i'.format(self.nnodevars),
            self.fp.read(self.nnodevars * self.RECORDSIZE))

        self.nlinkvars = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.vars[2] = struct.unpack(
            '{0}i'.format(self.nlinkvars),
            self.fp.read(self.nlinkvars * self.RECORDSIZE))

        self.vars[3] = [0]

        self.nsystemvars = struct.unpack('i', self.fp.read(self.RECORDSIZE))[0]
        self.vars[4] = struct.unpack(
            '{0}i'.format(self.nsystemvars),
            self.fp.read(self.nsystemvars * self.RECORDSIZE))

        # System vars do not have names per se, but made names = number labels
        self.names[4] = [self.varcode[4][i] for i in self.vars[4]]

        self.startdate = struct.unpack(
            'd', self.fp.read(2 * self.RECORDSIZE))[0]
        days = int(self.startdate)
        seconds = (self.startdate - days) * 86400
        self.startdate = datetime.datetime(1899, 12, 30) + \
            datetime.timedelta(days=days, seconds=seconds)

        self.reportinterval = struct.unpack(
            'i', self.fp.read(self.RECORDSIZE))[0]
        self.reportinterval = datetime.timedelta(
            seconds=self.reportinterval)

        # Calculate the bytes for each time period when
        # reading the computed results
        self.bytesperperiod = self.RECORDSIZE * (
            2 +
            self.swmm_nsubcatch * self.swmm_nsubcatchvars +
            self.swmm_nnodes * self.nnodevars +
            self.swmm_nlinks * self.nlinkvars +
            self.nsystemvars)

    def update_var_code(self, typenumber):
        start = len(self.varcode[typenumber])
        end = start + len(self.names[3])
        nlabels = list(range(start, end))
        ndict = dict(list(zip(nlabels, self.names[3])))
        self.varcode[typenumber].update(ndict)

    def type_check(self, itemtype):
        if itemtype in [0, 1, 2, 3, 4]:
            return itemtype
        try:
            typenumber = self.itemlist.index(itemtype)
        except ValueError:
            raise ValueError('''
*
*   Type argument "{0}" is incorrect.
*   Must be in "{1}".
*
'''.format(itemtype, list(range(5)) + self.itemlist))
        return typenumber

    def name_check(self, itemtype, itemname):
        self.itemtype = self.type_check(itemtype)
        try:
            itemindex = self.names[self.itemtype].index(itemname)
        except (ValueError, KeyError):
            raise ValueError('''
*
*   {0} was not found in "{1}" list.
*
'''.format(itemname, itemtype))
        return (itemname, itemindex)

    def get_swmm_results(self, itemtype, name, variableindex, period):
        if itemtype not in [0, 1, 2, 4]:
            raise ValueError('''
*
*   Type must be one of subcatchment (0), node (1). link (2), or system (4).
*   You gave "{0}".
*
'''.format(itemtype))

        _, itemindex = self.name_check(itemtype, name)

        date_offset = self.startpos + period * self.bytesperperiod

        self.fp.seek(date_offset, 0)
        date = struct.unpack('d', self.fp.read(2 * self.RECORDSIZE))[0]

        offset = date_offset + 2 * self.RECORDSIZE  # skip the date

        if itemtype == 0:
            offset = offset + self.RECORDSIZE * (
                itemindex * self.swmm_nsubcatchvars)
        if itemtype == 1:
            offset = offset + self.RECORDSIZE * (
                self.swmm_nsubcatch * self.swmm_nsubcatchvars +
                itemindex * self.nnodevars)
        elif itemtype == 2:
            offset = offset + self.RECORDSIZE * (
                self.swmm_nsubcatch * self.swmm_nsubcatchvars +
                self.swmm_nnodes * self.nnodevars +
                itemindex * self.nlinkvars)
        elif itemtype == 4:
            offset = offset + self.RECORDSIZE * (
                self.swmm_nsubcatch * self.swmm_nsubcatchvars +
                self.swmm_nnodes * self.nnodevars +
                self.swmm_nlinks * self.nlinkvars)

        offset = offset + self.RECORDSIZE * variableindex

        self.fp.seek(offset, 0)
        value = struct.unpack('f', self.fp.read(self.RECORDSIZE))[0]
        return (date, value)

    def get_dates(self):
        """Return start and end date tuple."""
        begindate = datetime.datetime(1899, 12, 30)
        ntimes = list(range(self.swmm_nperiods))
        periods = [ntimes[0], ntimes[-1]]
        st_end = []
        for period in periods:
            date_offset = self.startpos + period * self.bytesperperiod
            self.fp.seek(date_offset, 0)
            day = struct.unpack('d', self.fp.read(2 * self.RECORDSIZE))[0]
            st_end.append(begindate + datetime.timedelta(days=int(day)))
        return st_end


@mando.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def catalog(filename, itemtype='', tablefmt='simple', header='default'):
    """List the catalog of objects in output file

    Parameters
    ----------
    {filename}
    {itemtype}
    {tablefmt}
    {header}

    """
    obj = SwmmExtract(filename)
    if itemtype:
        typenumber = obj.type_check(itemtype)
        plist = [typenumber]
    else:
        plist = list(range(len(obj.itemlist)))
    if header == 'default':
        header = ['TYPE', 'NAME']
    collect = []
    for i in plist:
        for oname in obj.names[i]:
            collect.append([obj.itemlist[i], oname])
    return tsutils.printiso(collect,
                            tablefmt=tablefmt,
                            headers=header)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def listdetail(filename,
               itemtype,
               name='',
               tablefmt='simple',
               header='default'):
    """List nodes and metadata in output file.

    Parameters
    ----------
    {filename}
    {itemtype}
    name : str
        Optional specific name to print only that entry.  This can be
        looked up using 'listvariables'.
    {tablefmt}
    {header}

    """
    obj = SwmmExtract(filename)
    typenumber = obj.type_check(itemtype)
    if name:
        objectlist = [obj.name_check(itemtype, name)[0]]
    else:
        objectlist = obj.names[typenumber]

    propnumbers = obj.propcode[typenumber]
    if header == "default":
        header = ['#Name'] + [PROPCODE[typenumber][i] for i in propnumbers]

    collect = []
    for i, oname in enumerate(objectlist):
        printvar = [oname]
        for j in obj.prop[typenumber][i]:
            if j[0] == 0:
                printvar.append(TYPECODE[typenumber][j[1]])
            else:
                printvar.append(j[1])
        collect.append(printvar)
    df = pd.DataFrame(collect)
    cheader = []
    for head in header:
        if head not in cheader:
            cheader.append(head)
        else:
            cnt = cheader.count(head)
            cheader.append('{0}.{1}'.format(head, cnt))
    df.columns = cheader
    return tsutils.printiso(df,
                            tablefmt=tablefmt,
                            headers=header)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def listvariables(filename, tablefmt='csv_nos', header='default'):
    """List variables available for each type.

    The type are "subcatchment", "node", "link", "pollutant", "system".

    Parameters
    ----------
    {filename}
    {tablefmt}
    {header}

    """
    obj = SwmmExtract(filename)
    if header == 'default':
        header = ['TYPE', 'DESCRIPTION', 'VARINDEX']
    # 'pollutant' really isn't it's own itemtype
    # but part of subcatchment, node, and link...
    collect = []
    for itemtype in ['subcatchment', 'node', 'link', 'system']:
        typenumber = obj.type_check(itemtype)

        obj.update_var_code(typenumber)

        for i in obj.vars[typenumber]:
            try:
                collect.append([itemtype,
                                obj.varcode[typenumber][i].decode(),
                                i])
            except (TypeError, AttributeError):
                collect.append([itemtype,
                                str(obj.varcode[typenumber][i]),
                                str(i)])
    return tsutils.printiso(collect,
                            tablefmt=tablefmt,
                            headers=header)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def stdtoswmm5(start_date=None, end_date=None, input_ts='-'):
    """Take the toolbox standard format and return SWMM5 format.

    Toolbox standard::

       Datetime, Column_Name
       2000-01-01 00:00:00 ,  45.6
       2000-01-01 01:00:00 ,  45.2
       ...

    SWMM5 format::

        ; comment line
        01/01/2000 00:00, 45.6
        01/01/2000 01:00, 45.2
        ...

    Parameters
    ----------
    {input_ts}
    {start_date}
    {end_date}

    """
    import csv
    sys.tracebacklimit = 1000
    tsd = tsutils.read_iso_ts(input_ts)[start_date:end_date]
    try:
        # Header
        print(';Datetime,', ', '.join(str(i) for i in tsd.columns))

        # Data
        cols = tsd.columns.tolist()
        tsd['date_tmp_tstoolbox'] = tsd.index.format(formatter=lambda x:
                                                     x.strftime('%m/%d/%Y'))
        tsd['time_tmp_tstoolbox'] = tsd.index.format(formatter=lambda x:
                                                     x.strftime('%H:%M:%S'))
        tsd.to_csv(sys.stdout, float_format='%g', header=False, index=False,
                   cols=['date_tmp_tstoolbox', 'time_tmp_tstoolbox'] + cols,
                   sep=' ', quoting=csv.QUOTE_NONE)
    except IOError:
        return


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def getdata(filename, *labels):
    """DEPRECATED: Use 'extract' instead."""
    return extract(filename, *labels)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(_LOCAL_DOCSTRINGS)
def extract(filename, *labels):
    """Get the time series data for a particular object and variable.

    Parameters
    ----------
    {filename}
    {labels}

    """
    obj = SwmmExtract(filename)
    jtsd = []
    for label in labels:
        itemtype, name, variableindex = label.split(',')
        typenumber = obj.type_check(itemtype)
        # if itemtype != 'system':
        name = obj.name_check(itemtype, name)[0]

        obj.update_var_code(typenumber)

        begindate = datetime.datetime(1899, 12, 30)
        dates = []
        values = []
        for time in range(obj.swmm_nperiods):
            date, value = obj.get_swmm_results(
                typenumber, name, int(variableindex), time)
            days = int(date)
            seconds = int((date - days) * 86400)
            extra = seconds % 10
            if extra != 0:
                if extra == 9:
                    seconds = seconds + 1
                if extra == 1:
                    seconds = seconds - 1
            date = begindate + datetime.timedelta(
                days=days, seconds=seconds)
            dates.append(date)
            values.append(value)
        jtsd.append(pd.DataFrame(
            pd.Series(values, index=dates),
            columns=['{0}_{1}_{2}'.format(
                itemtype, name, obj.varcode[typenumber][int(variableindex)])]))
    result = pd.concat(jtsd, axis=1, join_axes=[jtsd[0].index])
    return tsutils.printiso(result)

def fast_extract(filename, *labels):
    """Get the time series data for a particular object and variable.

    Parameters
    ----------
    {filename}
    {labels}

    """
    obj = SwmmExtract(filename)
    begindate = datetime.datetime(1899, 12, 30)
    namemap = {
        0 : 'subcatchment',
        1 : 'node',
        2 : 'link',
        4 : 'system'
    }
    typemap = {
        0 : 0,
        1 : int(obj.RECORDSIZE * obj.swmm_nsubcatch * obj.swmm_nsubcatchvars),
        2 : int(obj.RECORDSIZE * obj.swmm_nsubcatch * obj.swmm_nsubcatchvars +
                obj.RECORDSIZE * obj.swmm_nnodes * obj.nnodevars),
        4 : int(obj.RECORDSIZE * obj.swmm_nsubcatch * obj.swmm_nsubcatchvars +
                obj.RECORDSIZE * obj.swmm_nnodes * obj.nnodevars +
                obj.RECORDSIZE * obj.swmm_nlinks * obj.nlinkvars),
    }
    varmap = {
        0 : obj.swmm_nsubcatchvars,
        1 : obj.nnodevars,
        2 : obj.nlinkvars,
        4 : obj.nsystemvars,
    }
    labels = pd.Series(labels).str.split(',')
    itypename = labels.str[0]
    item_names = labels.str[1].values
    varix = labels.str[2].astype(int)
    itypes = itypename.map({v: k for k, v in namemap.items()}).values
    itemindices = []
    for dtype, name in zip(itypes, item_names):
        _, itemindex = obj.name_check(dtype, name)
        itemindices.append(itemindex)
    item_name_map = pd.Series(item_names, index=itemindices)
    item_name_map = item_name_map[~item_name_map.index.duplicated(keep='first')]
    items = pd.np.asarray(itemindices)
    periods = pd.np.arange(obj.swmm_nperiods + 1)
    date_offsets = obj.startpos + periods * obj.bytesperperiod
    offsets = date_offsets + 2 * obj.RECORDSIZE
    type_offsets = pd.Series(itypes).map(typemap).values
    item_offsets = items * pd.Series(itypes).map(varmap).values * obj.RECORDSIZE
    var_offsets = varix * obj.RECORDSIZE
    all_offsets = type_offsets + item_offsets + var_offsets
    bool_mask = pd.np.zeros(obj.bytesperperiod - 2).astype(bool)
    record_mask = bool_mask.copy()
    record_mask[all_offsets] = 1
    for i in range(obj.RECORDSIZE):
        bool_mask[all_offsets + i] = 1
    ts = []
    for start_ix, end_ix in zip(periods[:-1], periods[:-1] + 1):
        mmap_offset = (mmap.ALLOCATIONGRANULARITY *
                       math.floor(offsets[start_ix] /
                                  mmap.ALLOCATIONGRANULARITY))
        mmap_endpoint = offsets[end_ix] - 2
        mmap_length = mmap_endpoint - mmap_offset
        assert((mmap_offset % mmap.ALLOCATIONGRANULARITY) == 0)
        mmap_start_gap = offsets[start_ix] - mmap_offset
        mmap_end_gap = mmap_endpoint - (offsets[end_ix] - 2)
        mmap_bool_mask = pd.np.pad(bool_mask, pad_width=(mmap_start_gap, mmap_end_gap),
                                   mode='constant', constant_values=0)
        mmap_fp = mmap.mmap(obj.fp.fileno(), prot=mmap.PROT_READ,
                            length=mmap_length, offset=mmap_offset)
        records = pd.np.array(mmap_fp)[mmap_bool_mask].reshape(-1, obj.RECORDSIZE)
        records = records.view(dtype=pd.np.float32).ravel()
        ts.append(records)
    dates = []
    for date in date_offsets[:-1]:
        obj.fp.seek(date, 0)
        date = struct.unpack('d', obj.fp.read(2 * obj.RECORDSIZE))[0]
        dates.append(date)
    record_order = (pd.DataFrame(pd.np.column_stack([itypes, items, varix]))
                    .sort_values(by=[0,1,2]))
    record_type_names = record_order[0].map(namemap).astype(str)
    record_item_names = record_order[1].map(item_name_map).astype(str)
    record_var_names = pd.concat([record_order[2][record_order[0] == i]
                                  .map(VARCODE[i]) for i in
                                VARCODE]).sort_index().astype(str)
    headings = record_type_names + '_' + record_item_names + '_' + record_var_names
    date_index = pd.to_datetime(dates, unit='d', origin=begindate)
    result = pd.DataFrame(pd.np.vstack(ts), index=date_index, columns=headings)
    return result

@tsutils.doc(_LOCAL_DOCSTRINGS)
def extract_arr(filename, *labels):
    """Same as extract except it returns the raw numpy array.

    Available only within Python API

    Parameters
    ----------
    {filename}
    {labels}

    """
    obj = SwmmExtract(filename)
    for label in labels:
        itemtype, name, variableindex = label.split(',')
        typenumber = obj.type_check(itemtype)
        if itemtype != 'system':
            name = obj.name_check(itemtype, name)[0]

        obj.update_var_code(typenumber)

        data = pd.np.zeros(len(list(range(obj.swmm_nperiods))))

        for time in range(obj.swmm_nperiods):
            _, value = obj.get_swmm_results(typenumber,
                                            name,
                                            int(variableindex),
                                            time)
            data[time] = value

    return data


def main():
    if not os.path.exists('debug_swmmtoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
