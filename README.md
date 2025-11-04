# MTG/MTFRPPixel Data Processor

Python script for downloading, processing, and analyzing MTG (Meteosat Third Generation) fire detection data from LSA SAF (Land Surface Analysis Satellite Applications Facility).

## 📋 Description

This script automates the workflow with MTFRPPixel (Fire Radiative Power) data from the MTG satellite, from download to creating QGIS-ready files. It includes spatial filtering, format conversion, and efficient processing of large data volumes.

## ✨ Features

- **Automatic Download**: Downloads data directly from LSA SAF servers using authentication
- **Temporal Flexibility**: Support for specific year, month, or day
- **Spatial Filtering**: Applies custom bounding box to reduce data volume
- **Efficient Processing**: Two modes (fast or low RAM usage)
- **QGIS Conversion**: Generates CSV and GeoPackage with spatial geometries
- **Memory Management**: Batch processing for large datasets
- **CLI Interface**: Flexible parameters via command line

## 🛠️ Installation

### Prerequisites

```bash
# Install Python dependencies
pip install pandas geopandas shapely

# On Ubuntu/Debian, install wget if needed
sudo apt-get install wget
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
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 08
```

**Download a specific day:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 08 --day 15
```

### Advanced Examples

**With custom bounding box:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --month 08 --bbox "35.0,-10.0,43.0,-5.0"
```

**With custom output name:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --output_name portugal_fires
```

**Reprocess existing data:**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --skip_download
```

**CSV only (no GeoPackage):**
```bash
python mtg_frp_processor.py --username YOUR_USER --password YOUR_PASSWORD --year 2025 --skip_geopackage
```

## 📊 Parameters

### Required Parameters
- `--username`: LSA SAF username
- `--password`: LSA SAF password  
- `--year`: Year in YYYY format (e.g., 2025)

### Optional Parameters
- `--month`: Month in MM format (e.g., 08)
- `--day`: Day in DD format (e.g., 15)
- `--base_dir`: Download directory (default: "FRP_MTG")
- `--bbox`: Bounding box "lat_min,lon_min,lat_max,lon_max" (default: Portugal)
- `--output_name`: Base name for output files
- `--skip_download`: Skip download, use existing data
- `--skip_geopackage`: Don't create GeoPackage, only CSV

## 🗂️ Output Structure

### Generated Files
- `MTG_MTFRPPixel_aggregated_YYYY[_MM][_DD].csv` - Aggregated data in CSV
- `MTG_MTFRPPixel_aggregated_YYYY[_MM][_DD].gpkg` - GeoPackage for QGIS

### Data Structure
Files contain all original columns plus:
- `source_file`: Source file
- `acquisition_date`: Acquisition date
- `acquisition_datetime`: Acquisition datetime (converted from ACQTIME)
- Spatial geometry (only in GeoPackage)

## 🔧 Processing

### Processing Modes

1. **Standard Processing**: Faster, higher RAM usage
2. **Batch Processing**: Slower, optimized RAM usage

### Applied Filters

- **Spatial Filter**: Only data within bounding box
- **Temporal Filter**: Data from specified period
- **Validation**: Data integrity verification

## 💡 Usage Tips

### For Large Data Volumes
```bash
# Use batch processing for complete years
python mtg_frp_processor.py --username USER --password PASS --year 2025 --skip_geopackage
```

### For Regional Analysis
```bash
# Bounding box for mainland Portugal (default)
--bbox "36.87164804628416,-9.633111264309846,42.24431922230131,-6.070242597727865"

# Bounding box for Iberian Peninsula
--bbox "35.0,-10.0,44.0,-3.0"
```

### QGIS Integration
1. **GeoPackage**: Open directly as vector layer
2. **CSV**: Import as delimited text layer
   - X field: LONGITUDE
   - Y field: LATITUDE  
   - CRS: EPSG:4326

## 🐛 Troubleshooting

### Common Issues

**Download fails:**
- Check LSA SAF credentials
- Verify network connectivity
- Confirm period exists on server

**Memory issues:**
- Use `--skip_geopackage` to reduce processing
- Choose batch processing when prompted
- Process by months instead of complete years

**Date parsing errors:**
- Data maintains original ACQTIME format
- Script automatically converts to datetime

### Logs and Debug
Script provides detailed feedback on:
- Download progress
- Number of files processed
- Filtered data statistics
- Memory usage

## 📝 Complete Workflow Example

```bash
# 1. Download and process one month
python mtg_frp_processor.py --username user --password pass --year 2025 --month 07 --output_name july_2025

# 2. Open in QGIS
# File: july_2025_2025_07.gpkg

# 3. Analyze specific data
# Filter by FRP (Fire Radiative Power)
# FRP > 10 for significant fires
```

## 🔒 Security

- Credentials passed only via command line
- Connections use SSL/TLS
- Temporary data automatically deleted

## 📄 License

This project is distributed under the MIT License. See LICENSE file for details.

## 🙋 Support

For issues and questions:
1. Check troubleshooting section
2. Create GitHub issue
3. Contact project maintainer

## 📚 References

- [LSA SAF Product Documentation](https://lsa-saf.eumetsat.int)
- [MTG Mission Overview](https://www.eumetsat.int/mtg)
- [QGIS Documentation](https://qgis.org)

---

**Note**: Requires valid LSA SAF credentials. Register at [https://lsa-saf.eumetsat.int](https://lsa-saf.eumetsat.int).
