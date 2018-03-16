import datetime
import mmap
import math
import struct
import pandas as pd
import numpy as np
from swmmtoolbox import swmmtoolbox as swmm

obj = swmm.SwmmExtract('tests/frutal.out')
self = obj

c = swmm.catalog('tests/frutal.out')
c = pd.DataFrame(c)
item_names = c[c[0] == 'node'][1].values

itemindices = []

for name in item_names:
    _, itemindex = self.name_check(1, name)
    itemindices.append(itemindex)

item_name_map = pd.Series(item_names, index=itemindices)

items = np.repeat(itemindices, 3)
varix = np.tile([0, 1, 4], len(itemindices))
itypes = np.repeat(1, len(items))

nnoderecords = self.swmm_nnodes * self.nnodevars

periods = np.arange(self.swmm_nperiods + 1)
date_offsets = self.startpos + periods * self.bytesperperiod
offsets = date_offsets + 2 * self.RECORDSIZE

namemap = {
    0 : 'subcatchment',
    1 : 'node',
    2 : 'link',
    4 : 'system'
}

typemap = {
    0 : 0,
    1 : int(self.RECORDSIZE * self.swmm_nsubcatch * self.swmm_nsubcatchvars),
    2 : int(self.RECORDSIZE * self.swmm_nsubcatch * self.swmm_nsubcatchvars +
            self.RECORDSIZE * self.swmm_nnodes * self.nnodevars),
    4 : int(self.RECORDSIZE * self.swmm_nsubcatch * self.swmm_nsubcatchvars +
            self.RECORDSIZE * self.swmm_nnodes * self.nnodevars +
            self.RECORDSIZE * self.swmm_nlinks * self.nlinkvars),
}

varmap = {
    0 : self.swmm_nsubcatchvars,
    1 : self.nnodevars,
    2 : self.nlinkvars,
    4 : self.nsystemvars,
}

type_offsets = pd.Series(itypes).map(typemap).values
item_offsets = items * pd.Series(itypes).map(varmap).values * self.RECORDSIZE
var_offsets = varix * self.RECORDSIZE
all_offsets = type_offsets + item_offsets + var_offsets

bool_mask = np.zeros(self.bytesperperiod - 2).astype(bool)
record_mask = bool_mask.copy()
record_mask[all_offsets] = 1
for i in range(self.RECORDSIZE):
    bool_mask[all_offsets + i] = 1

ts = []

for start_ix, end_ix in zip(periods[:-1], periods[:-1] + 1):
    print(start_ix)
    mmap_offset = (mmap.PAGESIZE * math.floor(offsets[start_ix] / mmap.PAGESIZE))
    mmap_endpoint = offsets[end_ix] - 2
    mmap_length = mmap_endpoint - mmap_offset
    assert((mmap_offset % mmap.ALLOCATIONGRANULARITY) == 0)
    mmap_start_gap = offsets[start_ix] - mmap_offset
    mmap_end_gap = mmap_endpoint - (offsets[end_ix] - 2)
    mmap_bool_mask = np.pad(bool_mask, pad_width=(mmap_start_gap, mmap_end_gap), mode='constant',
                            constant_values=0)
    mmap_fp = mmap.mmap(self.fp.fileno(), prot=mmap.PROT_READ,
                        length=mmap_length, offset=mmap_offset)
    records = np.array(mmap_fp)[mmap_bool_mask].reshape(-1, self.RECORDSIZE)
    records = records.view(dtype=np.float32).ravel()
    ts.append(records)

dates = []
for date in date_offsets[:-1]:
    self.fp.seek(date, 0)
    date = struct.unpack('d', self.fp.read(2 * self.RECORDSIZE))[0]
    dates.append(date)

record_order = pd.DataFrame(np.column_stack([itypes, items, varix])).sort_values(by=[0,1,2])

record_type_names = record_order[0].map(namemap).astype(str)
record_item_names = record_order[1].map(item_name_map).astype(str)
record_var_names = pd.concat([record_order[2][record_order[0] == i].map(swmm.VARCODE[i]) for i in
                              swmm.VARCODE]).sort_index().astype(str)
headings = record_type_names + '_' + record_item_names + '_' + record_var_names

begindate = datetime.datetime(1899, 12, 30)
date_index = pd.to_datetime(dates, unit='d', origin=begindate)

data = pd.DataFrame(np.vstack(ts), index=date_index, columns=headings)
