import os
import glob
import re
import intake

class SHIFTCatalog():
    def __init__(self):
        self.dir = "/efs/efs-data-curated/v1"
        self.catalog_name = "shift_catalog.yml"
        self.catalog = intake.open_catalog(os.path.join(self.dir, self.catalog_name))
        
        for dataset in list(self.catalog):
            setattr(self, dataset, getattr(self, dataset))
            
            try:
                child = list(getattr(self, dataset))
                setattr(getattr(self, dataset), 'datasets', child)
            except:
                pass
            
        dates = sorted([d for d in os.listdir( self.dir) if re.search("^\d{8}$", d)])
       
        exp =  "(?<!\d)\d{6}(?!\d)"
        data = {}
        for date in dates:
            files = glob.glob(os.path.join(self.dir, date, 'L2a', "*.hdr"))
            times = sorted([re.search(exp, f).group() for f in files if re.search(exp, f)])
            data[date] = times

        setattr(self, 'dates', list(data.keys()))
        setattr(self, 'times', data)
                    
    
    def __getattr__(self, name):
        return getattr(self.catalog, name)
    
    @property
    def datasets(self):
        return list(self.catalog)