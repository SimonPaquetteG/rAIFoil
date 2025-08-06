#include "help.h"

// define it as one big raw‐string literal (C++11+)
const char help_text[] = R"HELP(
1. Expected geometry format

  a. Selig format: x,y format for points starting from trailing edge, along the upper surface to the leading edge and back around the lower surface to trailing edge
  b. x ≈ 0 at leading edge
  c. x = 1 at trailing edge
  d. If geometry does not close at the trailing edge, remove any additional point(s) at x = 1 between upper and lower surfaces that form the trailing edge
  e. No consecutive points should have both the same x,y coordinate values

2. Expected input file format

  a. Geometry

    # Either GEOMETRY OR GEOMETRY_FOLDER (mutually exclusive)

    GEOMETRY_FOLDER = # Folder containing geometry files in Selig format
    GEOMETRY        = # Geometry file in Selig format

  b. Flow Conditions

    AOA_MIN         = # Min value for Angle of Attack input range (degrees)
    AOA_MAX         = # Max value for Angle of Attack input range (degrees)
    AOA_STEP        = # Sampling interval for Angle of Attack input range (degrees)

    MACH_MIN        = # Min value for Mach number input range
    MACH_MAX        = # Max value for Mach number input range
    MACH_STEP       = # Sampling interval for Mach number input range

    REYNOLDS_MIN    = # Min value for Reynolds number input range
    REYNOLDS_MAX    = # Max value for Reynolds number input range
    REYNOLDS_STEP   = # Sampling interval for Reynolds number input range

  c. Outputs

    CI_LEVEL        = # Confidence level (CI: 0 to 1, CI != 1) for output features (not available in c81 file format)

    OUTPUT_FILE     = # Output file name prefix
    OUTPUT_FOLDER   = # Output folder name
    OUTPUT_FORMAT   = # Format for output files (csv or c81)
    
)HELP";
