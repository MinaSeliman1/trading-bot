"""
A fixed, documented reference list of S&P 500 constituents, compiled from
general knowledge of index membership circa 2024-2025 -- NOT a live index
query (no data vendor for official constituents is wired into this
project), and NOT curated or filtered based on any backtest result. Used
purely as an objective sampling pool for independent-universe validation.
Sector groupings are for readability only; not used in any calculation.
"""

SP500_POOL = sorted(set([
    # Technology
    "ADI", "AMAT", "ANSS", "APH", "AVGO", "CDNS", "CDW", "CTSH", "ENPH", "FTNT",
    "GLW", "HPQ", "INTU", "JNPR", "KEYS", "KLAC", "LRCX", "MCHP", "MPWR", "MSI",
    "MU", "NOW", "NTAP", "NXPI", "ON", "PANW", "PTC", "QCOM", "ROP", "SNPS",
    "STX", "SWKS", "TDY", "TEL", "TER", "TXN", "TYL", "WDC", "ZBRA",
    # Healthcare
    "A", "ABT", "ALGN", "AMGN", "BAX", "BDX", "BIIB", "BIO", "BMY", "BSX",
    "CAH", "CI", "CNC", "COO", "CRL", "CTLT", "CVS", "DGX", "DHR", "DVA",
    "DXCM", "ELV", "EW", "GILD", "HCA", "HOLX", "HSIC", "HUM", "IDXX", "ILMN",
    "INCY", "IQV", "ISRG", "LH", "LLY", "MCK", "MDT", "MOH", "MRNA", "MTD",
    "PODD", "REGN", "RMD", "SYK", "TECH", "TFX", "TMO", "UHS", "VRTX", "VTRS",
    "WAT", "WST", "XRAY", "ZBH", "ZTS",
    # Financials
    "AFL", "AIG", "AJG", "ALL", "AMP", "AON", "BEN", "BK", "BLK", "BRO",
    "C", "CB", "CBOE", "CFG", "CINF", "CME", "COF", "DFS", "ERIE", "FDS",
    "FIS", "FITB", "GL", "GPN", "HBAN", "ICE", "IVZ", "JKHY", "KEY", "L",
    "MCO", "MET", "MKTX", "MMC", "MSCI", "MTB", "NDAQ", "NTRS", "PFG", "PGR",
    "PNC", "PRU", "RF", "RJF", "SCHW", "SPGI", "STT", "SYF", "TFC", "TROW",
    "TRV", "USB", "WRB", "WTW",
    # Consumer / Retail
    "AZO", "BBY", "BBWI", "CCL", "CHD", "CHRW", "CL", "CLX", "COST", "CPRT",
    "CTAS", "DG", "DHI", "DLTR", "DPZ", "EBAY", "ETSY", "EXPE", "GIS", "GRMN",
    "HAS", "HLT", "HSY", "IP", "K", "KDP", "KHC", "KMB", "KMX", "KR",
    "LEN", "LKQ", "LOW", "LULU", "LVS", "LYV", "MAR", "MDLZ", "MGM", "MHK",
    "MKC", "MNST", "NCLH", "NVR", "NWL", "ORLY", "PHM", "PM", "POOL", "PPL",
    "RCL", "RL", "ROL", "ROST", "SBUX", "SJM", "STLD", "STZ", "SYY", "TGT",
    "TJX", "TPR", "TSCO", "ULTA", "VFC", "WBA", "WHR", "WYNN", "YUM",
    # Industrials
    "ALLE", "AME", "AOS", "AXON", "CMI", "CSX", "DOV", "EFX", "EMR", "ETN",
    "EXPD", "FAST", "GD", "GNRC", "GWW", "HII", "HWM", "IEX", "IR", "ITW",
    "J", "JBHT", "JCI", "LDOS", "LHX", "MAS", "NDSN", "NOC", "NSC", "ODFL",
    "OTIS", "PCAR", "PH", "PWR", "RHI", "RSG", "SNA", "SWK", "TDG", "TT",
    "URI", "VRSK", "WAB", "WM", "XYL",
    # Energy
    "BKR", "CTRA", "EOG", "EQT", "FANG", "HAL", "HES", "KMI", "MPC", "OKE",
    "OXY", "PSX", "SLB", "TRGP", "VLO", "WMB",
]))
