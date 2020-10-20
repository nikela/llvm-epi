// Not ideal but we want to preserve the existing schedules as 
// undisturbed as possible.


def : ReadAdvance<ReadVPUFALU,       0>;
def : ReadAdvance<ReadVPUIALU,       0>;
def : ReadAdvance<ReadVPUScalarFALU,       0>;
def : ReadAdvance<ReadVPUScalarIALU,       0>;
def : ReadAdvance<ReadVPUStoreData,       0>;

def : WriteRes<WriteVPUFALU, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUIALU, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPULoadIndexed, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPULoad, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPULoadStrided, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUScalarFALU, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUScalarIALU, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUStoreIndexed, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUStore, []>  { let Unsupported = 1; }
def : WriteRes<WriteVPUStoreStrided, []>  { let Unsupported = 1; }
