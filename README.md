# MTG/MTFRPPixel Data Processor

Python script for downloading, processing, and analyzing MTG (Meteosat Third Generation) fire detection data from LSA SAF (Land Surface Analysis Satellite Applications Facility).

<img width="1450" height="851" alt="image" src="https://github.com/user-attachments/assets/2b9e6575-4459-43ed-9d6d-913d3e439d32" />


## 📋 Description

This script automates the workflow with MTFRPPixel (Fire Radiative Power) data from the MTG satellite, from download to creating QGIS-ready files. It includes spatial filtering, format conversion, and efficient processing of large data volumes.

## ✨ Features

- **Cross-Platform Download**: Downloads data using requests library (works on Windows, macOS, Linux)
- **In-Memory Processing**: Optional direct download to memory without saving intermediate files
- **Temporal Flexibility**: Support for a specific year, month, or day, as well as ranges/lists of months (`--months 07-08`) or days (`--days 26-28`, `--days 26,27,31`) — flexible formatting (8 or 08)
- **Spatial Filtering**: Applies custom bounding box (using parallax-corrected coordinates) to reduce data volume
- **Efficient Processing**: Two modes (fast or low RAM usage)
- **QGIS Conversion**: Generates CSV and GeoPackage with spatial geometries
- **Memory Management**: Batch processing for large datasets
- **CLI Interface**: Flexible parameters via command line
- **Data Validation**: Automatic year validation (≥2025) and date normalization

## 🛠️ Installation

### Prerequisites

```bash
# Install Python dependencies
pip install pandas geopandas shapely requests
```

### Download Script

```bash
git clone https://github.com/PedroVenancio/mtg-frp-processor.git
cd mtg-frp-processor
```

## 🚀 Usage

### Basic Examples

**Download a complete year:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025
```

**Download a specific month:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 8
# or
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 08
```

**Download a specific day:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 8 --day 5
# or  
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 08 --day 05
```

**Download a range or list of days within a single month:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 07 --days 26-28
# or a non-contiguous list
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 07 --days 26,27,31
```

**Download a range or list of months:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --months 07-08
# or
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --months 07,08,12
```

### Advanced Examples

**In-memory processing:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 8 --in_memory
```

**With custom bounding box:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 8 --bbox "35.0,-10.0,43.0,-5.0"
```

**With custom output name:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --output_name portugal_fires
```

**Reprocess existing data (no credentials needed):**
```bash
python mtg_frp_processor.py --year 2025 --skip_download
```

**CSV only (no GeoPackage):**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --skip_geopackage
```

**Process multiple months together:**
```bash
# Downloads both months to same directory, then processes entire year
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --base_dir my_data --skip_download
# Or simply use --months to do this in one command (see Advanced Examples above)
```

## 📊 Parameters

### Required Parameters
- `--username`: LSA SAF username (required unless `--skip_download` is used)
- `--password`: LSA SAF password (required unless `--skip_download` is used)
- `--year`: Year in YYYY format (must be ≥2025)

### Optional Parameters
- `--month`: Single month in MM format (1-12 or 01-12)
- `--months`: Multiple months as a range (e.g. `07-08`) or list (e.g. `07,08,12`). Cannot be combined with `--month`
- `--day`: Single day in DD format (1-31 or 01-31)
- `--days`: Multiple days as a range (e.g. `26-28`) or list (e.g. `26,27,31`). Requires exactly one month (via `--month` or a single-value `--months`). Cannot be combined with `--day`
- `--base_dir`: Download directory (default: "FRP_MTG")
- `--bbox`: Bounding box "lat_min,lon_min,lat_max,lon_max" (default: Portugal)
- `--output_name`: Base name for output files
- `--skip_download`: Skip download, use existing data (no credentials needed)
- `--skip_geopackage`: Don't create GeoPackage, only CSV
- `--in_memory`: Download directly to memory (no temporary files)

## 🗂️ Output Structure

### Generated Files
- `MTG_MTFRPPixel_aggregated_YYYY[_MM][_DD].csv` - Aggregated data in CSV
- `MTG_MTFRPPixel_aggregated_YYYY[_MM][_DD].gpkg` - GeoPackage for QGIS

### Data Structure
Files contain all original columns plus:
- `source_file`: Source file
- `acquisition_date`: Acquisition date
- `acquisition_slot_time`: Nominal acquisition slot (datetime), extracted from the filename timestamp (`YYYYMMDDHHMM`); MTG scans every 10 minutes, so this is one of 144 fixed slots per day
- `acquisition_datetime`: Actual acquisition datetime (converted from ACQTIME); can differ from `acquisition_slot_time` by a few minutes/seconds
- Spatial geometry (only in GeoPackage)

**Note on coordinates:** spatial filtering (bounding box) and the GeoPackage geometry both use the parallax-corrected coordinates (`LATITUDE_PARALLAX`, `LONGITUDE_PARALLAX`) rather than the raw `LATITUDE`/`LONGITUDE` fields, which account for the fire pixel's actual ground position after correcting for MTG's viewing angle. Both the raw and parallax-corrected columns are still kept in the output.

## 🔧 Processing

### Processing Modes

1. **Standard Processing**: Faster, higher RAM usage
2. **Batch Processing**: Slower, optimized RAM usage  
3. **In-Memory Processing**: Direct download to memory (most efficient)

### Applied Filters

- **Spatial Filter**: Only data within bounding box (using parallax-corrected coordinates)
- **Temporal Filter**: Data from specified period
- **Validation**: Data integrity verification and date normalization

### Multiple Periods (`--months`/`--days`)

When a range or list is given, each resulting month/day is downloaded and processed as its own period, then combined into a single output CSV/GeoPackage. As a safety net, exact-duplicate records (same source file, acquisition time, and coordinates) picked up across periods are automatically removed before saving.

## 💡 Usage Tips

### For Large Data Volumes

**For complete years or multiple months:**
```bash
# Use batch processing to manage memory
python mtg_frp_processor.py --username USER --password PASS --year 2025 --skip_geopackage
# When prompted, choose option 2 (batch processing)
```

**For single months or when you have ample RAM:**
```bash
# Use in-memory processing for faster results
python mtg_frp_processor.py --username USER --password PASS --year 2025 --month 8 --in_memory
```

**Alternative: Process by months and combine later:**
```bash
# Process months separately
python mtg_frp_processor.py --username USER --password PASS --year 2025 --month 7 --output_name summer_part1
python mtg_frp_processor.py --username USER --password PASS --year 2025 --month 8 --output_name summer_part2

# Combine CSVs manually if needed
```

### Memory Management Guidelines

- **1-3 months**: Safe for in-memory processing (16GB+ RAM)
- **4-6 months**: Use standard processing with batch option  
- **Complete year**: Always use batch processing
- **Uncertain**: Start with batch processing, monitor RAM usage


### For Regional Analysis
```bash
# Bounding box for mainland Portugal (default)
--bbox "36.87164804628416,-9.633111264309846,42.24431922230131,-6.070242597727865"

# Bounding box for Iberian Peninsula
--bbox "35.0,-10.0,44.0,-3.0"
```

### Processing Multiple Periods
The simplest way is to use `--months`/`--days` directly (see Advanced Examples above), which downloads and combines the periods automatically:
```bash
python mtg_frp_processor.py --username USER --password PASS --year 2025 --months 07-08
```

Alternatively, you can still download months separately into the same base directory and process them together afterwards:
```bash
# Download months separately to same base directory
python mtg_frp_processor.py --username USER --password PASS --year 2025 --month 7 --base_dir my_data
python mtg_frp_processor.py --username USER --password PASS --year 2025 --month 8 --base_dir my_data

# Then process entire year together
python mtg_frp_processor.py --year 2025 --base_dir my_data --skip_download
```

### QGIS Integration
1. **GeoPackage**: Open directly as vector layer
2. **CSV**: Import as delimited text layer
   - X field: LONGITUDE_PARALLAX
   - Y field: LATITUDE_PARALLAX
   - CRS: EPSG:4326

## 🐛 Troubleshooting

### Common Issues

**Download fails:**
- Check LSA SAF credentials
- Verify network connectivity
- Confirm period exists on server (data available from 2025 onwards)

**Memory issues:**
- Use `--skip_geopackage` to reduce processing
- Choose batch processing when prompted
- Process by months instead of complete years
- Use `--in_memory` for most efficient processing

**Date format errors:**
- Script automatically normalizes month/day formats (8 → 08)
- Data maintains original ACQTIME format
- Script automatically converts to datetime

### Logs and Debug
Script provides detailed feedback on:
- Download progress and file discovery
- Number of files processed
- Filtered data statistics
- Memory usage and processing mode

## 📝 Complete Workflow Example

```bash
# 1. Download and process one month using in-memory processing
python mtg_frp_processor.py --username user --password pass --year 2025 --month 7 --output_name july_2025 --in_memory

# 2. Open in QGIS
# File: july_2025_2025_07.gpkg

# 3. Analyze specific data
# Filter by FRP (Fire Radiative Power)
# FRP > 10 for significant fires

# 4. Process multiple months together
python mtg_frp_processor.py --year 2025 --base_dir annual_data --skip_download --output_name summer_2025
```

## 🔒 Security

- Credentials passed only via command line
- Connections use SSL/TLS
- Temporary data automatically deleted
- No persistent storage of credentials

## 📄 License

This project is distributed under the MIT License. See LICENSE file for details.

## 🙋 Support

For issues and questions:
1. Check troubleshooting section
2. Create GitHub issue
3. Contact project maintainer

## 📚 References

- [LSA SAF Product Documentation](https://lsa-saf.eumetsat.int/en/news/news/2025/10/01/release-of-mtg-fire-radiative-power-product-as-demonstration/)
- [MTG Mission Overview](https://www.eumetsat.int/meteosat-third-generation)
- [QGIS Documentation](https://qgis.org/resources/hub/)

---

**Notes**: 
- Requires valid LSA SAF credentials. Register at [https://lsa-saf.eumetsat.int](https://datalsasaf.lsasvcs.ipma.pt/).
- Data is only available from January 1st, 2025 onwards.
- Script automatically normalizes date formats (8 → 08, 5 → 05) for compatibility.