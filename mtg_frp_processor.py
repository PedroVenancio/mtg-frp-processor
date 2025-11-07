import os
import gzip
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
import argparse
import sys
import requests
from io import BytesIO
from base64 import b64encode
import calendar
from urllib.parse import urljoin

def parse_bbox(bbox_str):
    """
    Parse bounding box string into coordinates
    Format: "lat_min,lon_min,lat_max,lon_max" or "lat_min,lon_min;lat_max,lon_max"
    """
    try:
        # Replace common separators with commas
        bbox_str = bbox_str.replace(';', ',').replace(' ', '')
        coords = [float(coord) for coord in bbox_str.split(',')]
        
        if len(coords) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        
        lat_min, lon_min, lat_max, lon_max = coords
        
        # Validate coordinates
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if lat_min >= lat_max:
            raise ValueError("lat_min must be less than lat_max")
        if lon_min >= lon_max:
            raise ValueError("lon_min must be less than lon_max")
        
        return lat_min, lon_min, lat_max, lon_max
    
    except Exception as e:
        print(f"Error parsing bounding box: {e}")
        print("Please use format: lat_min,lon_min,lat_max,lon_max")
        sys.exit(1)

def build_url(year, month=None, day=None):
    """
    Build the download URL based on the specified time period
    """
    base_url = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MTG/MTFRPPixel/NATIVE"
    
    if day and month:
        return f"{base_url}/{year}/{month}/{day}/"
    elif month:
        return f"{base_url}/{year}/{month}/"
    else:
        return f"{base_url}/{year}/"

def generate_file_urls_for_period(year, month=None, day=None):
    """
    Generate all possible file URLs for a given period
    """
    base_pattern = "LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_{timestamp}.csv.gz"
    urls = []
    
    if day and month:
        # For specific day: generate URLs for all hours and 10-minute intervals
        for hour in range(24):
            for minute in range(0, 60, 10):
                timestamp = f"{year}{month}{day}{hour:02d}{minute:02d}"
                filename = base_pattern.format(timestamp=timestamp)
                url = urljoin(build_url(year, month, day), filename)
                urls.append(url)
    
    elif month:
        # For month: generate URLs for all days in the month
        _, num_days = calendar.monthrange(int(year), int(month))
        for day_num in range(1, num_days + 1):
            day_str = f"{day_num:02d}"
            for hour in range(24):
                for minute in range(0, 60, 10):
                    timestamp = f"{year}{month}{day_str}{hour:02d}{minute:02d}"
                    filename = base_pattern.format(timestamp=timestamp)
                    url = urljoin(build_url(year, month, day_str), filename)
                    urls.append(url)
    
    else:
        # For year: generate URLs for all months and days
        for month_num in range(1, 13):
            month_str = f"{month_num:02d}"
            _, num_days = calendar.monthrange(int(year), month_num)
            for day_num in range(1, num_days + 1):
                day_str = f"{day_num:02d}"
                for hour in range(24):
                    for minute in range(0, 60, 10):
                        timestamp = f"{year}{month_str}{day_str}{hour:02d}{minute:02d}"
                        filename = base_pattern.format(timestamp=timestamp)
                        url = urljoin(build_url(year, month_str, day_str), filename)
                        urls.append(url)
    
    return urls

def check_existing_download(base_dir, year, month=None, day=None):
    """
    Check if download directory exists and has data for the specified period
    """
    if day and month:
        period_dir = Path(base_dir) / "PRODUCTS" / "MTG" / "MTFRPPixel" / "NATIVE" / year / month / day
    elif month:
        period_dir = Path(base_dir) / "PRODUCTS" / "MTG" / "MTFRPPixel" / "NATIVE" / year / month
    else:
        period_dir = Path(base_dir) / "PRODUCTS" / "MTG" / "MTFRPPixel" / "NATIVE" / year

    if not period_dir.exists():
        return False

    # Check if there are any .gz files in the period directory
    gz_files = list(period_dir.rglob("*.gz"))
    return len(gz_files) > 0

def download_with_requests(url, username, password):
    """
    Download file using requests (cross-platform)
    """
    try:
        auth_header = f"Basic {b64encode(f'{username}:{password}'.encode()).decode()}"
        headers = {'Authorization': auth_header}
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        # Don't print error for 404 (file not found) - this is normal for generated URLs
        if hasattr(e.response, 'status_code') and e.response.status_code == 404:
            return None
        print(f"Download failed for {url}: {e}")
        return None

def download_data_requests(year, month=None, day=None, username=None, password=None, base_dir="FRP_MTG"):
    """
    Download MTFRPPixel data using requests for the specified period
    """
    Path(base_dir).mkdir(exist_ok=True)
    
    # Build period string for logging
    period_str = f"{year}"
    if month:
        period_str += f"-{month}"
    if day:
        period_str += f"-{day}"
        
    print(f"Downloading data for {period_str}...")
    
    # Generate all possible file URLs for the period
    file_urls = generate_file_urls_for_period(year, month, day)
    
    if not file_urls:
        print(f"No file URLs generated for the specified period.")
        return False
    
    print(f"Trying {len(file_urls)} potential URLs...")
    
    success_count = 0
    for i, file_url in enumerate(file_urls):
        filename = os.path.basename(file_url)
        
        # Determine local path based on URL structure
        relative_path = file_url.replace("https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/", "")
        local_path = Path(base_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists
        if local_path.exists():
            success_count += 1
            if (i + 1) % 100 == 0:  # Print progress every 100 files
                print(f"Checked {i + 1}/{len(file_urls)} URLs... Found {success_count} files so far")
            continue
        
        content = download_with_requests(file_url, username, password)
        if content is not None:
            with open(local_path, 'wb') as f:
                f.write(content)
            success_count += 1
        
        if (i + 1) % 100 == 0:  # Print progress every 100 files
            print(f"Processed {i + 1}/{len(file_urls)} URLs... Downloaded {success_count} files so far")
    
    print(f"Download completed! {success_count}/{len(file_urls)} files downloaded successfully.")
    return success_count > 0

def process_in_memory(year, month=None, day=None, username=None, password=None, 
                     bbox_coords=None, output_filename=None):
    """
    Download and process data directly to memory without saving to disk
    """
    if bbox_coords is None:
        bbox_coords = (36.87164804628416, -9.633111264309846, 42.24431922230131, -6.070242597727865)
    
    # Generate all possible file URLs for the period
    urls = generate_file_urls_for_period(year, month, day)
    
    if not urls:
        print("No URLs generated for the specified period.")
        return None
    
    all_dataframes = []
    processed_count = 0
    total_records = 0
    downloaded_count = 0
    
    lat_min, lon_min, lat_max, lon_max = bbox_coords
    print(f"Downloading and processing directly to memory...")
    print(f"Trying {len(urls)} potential URLs...")
    print(f"Applying bounding box filter: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°W to {lon_max:.2f}°W")
    
    for i, url in enumerate(urls):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(urls)} URLs... Found {downloaded_count} files with data so far")
        
        content = download_with_requests(url, username, password)
        if content is None:
            continue
            
        downloaded_count += 1
            
        try:
            # Decompress and read directly to pandas
            with gzip.GzipFile(fileobj=BytesIO(content)) as f:
                df = pd.read_csv(f, low_memory=False)
            
            # Apply bounding box filter
            df_filtered = filter_by_bbox(df, bbox_coords)
            
            if not df_filtered.empty:
                df_filtered = df_filtered.copy()
                df_filtered.loc[:, 'source_file'] = os.path.basename(url)
                
                # Extract date from filename
                try:
                    # Extract timestamp from filename: LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_202508041200.csv.gz
                    timestamp_str = url.split('_')[-1].split('.')[0]
                    if len(timestamp_str) == 12:  # YYYYMMDDHHMM
                        year_file = timestamp_str[:4]
                        month_file = timestamp_str[4:6]
                        day_file = timestamp_str[6:8]
                        hour_file = timestamp_str[8:10]
                        minute_file = timestamp_str[10:12]
                        df_filtered.loc[:, 'acquisition_date'] = f"{year_file}-{month_file}-{day_file}"
                        
                        # Create acquisition_datetime from components
                        acq_datetime = datetime(
                            int(year_file), int(month_file), int(day_file),
                            int(hour_file), int(minute_file)
                        )
                        df_filtered.loc[:, 'acquisition_datetime'] = acq_datetime
                except Exception as e:
                    print(f"Warning: Could not parse date from filename {url}: {e}")
                    df_filtered.loc[:, 'acquisition_date'] = f"{year}-{month or '01'}-{day or '01'}"
                
                all_dataframes.append(df_filtered)
                processed_count += 1
                total_records += len(df_filtered)
                
                if downloaded_count % 10 == 0:
                    print(f"  Downloaded {downloaded_count} files, found {total_records} records so far")
        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
        
        # Clear memory periodically
        if (i + 1) % 50 == 0:
            import gc
            gc.collect()
    
    if not all_dataframes:
        print("No data was successfully processed.")
        return None
    
    print(f"Combining {len(all_dataframes)} filtered datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Saving combined data to {output_filename}...")
    combined_df.to_csv(output_filename, index=False)
    
    print(f"\nIn-memory processing complete!")
    print(f"Total URLs tried: {len(urls)}")
    print(f"Total files found: {downloaded_count}")
    print(f"Total files with data: {processed_count}")
    print(f"Total records after filtering: {len(combined_df)}")
    
    # Show datetime range if available
    if 'acquisition_datetime' in combined_df.columns:
        min_dt = combined_df['acquisition_datetime'].min()
        max_dt = combined_df['acquisition_datetime'].max()
        print(f"Datetime range: {min_dt} to {max_dt}")
    
    return output_filename

def filter_by_bbox(df, bbox_coords):
    """
    Filter DataFrame by bounding box to reduce memory usage
    """
    lat_min, lon_min, lat_max, lon_max = bbox_coords
    mask = (
        (df['LATITUDE'] >= lat_min) & 
        (df['LATITUDE'] <= lat_max) & 
        (df['LONGITUDE'] >= lon_min) & 
        (df['LONGITUDE'] <= lon_max)
    )
    return df[mask].copy()

def parse_acqtime(acqtime_str):
    """
    Convert ACQTIME string (YYYYMMDDHHMMSS) to datetime object
    """
    try:
        # Convert string to datetime
        return datetime.strptime(str(acqtime_str), '%Y%m%d%H%M%S')
    except (ValueError, TypeError):
        return None

def process_single_file(gz_file, bbox_coords):
    """
    Process a single .gz file with bounding box filtering
    Returns filtered DataFrame or None if error
    """
    try:
        # Extract date from file path: .../year/month/day/file.gz
        path_parts = Path(gz_file).parts
        
        # Find the indices for year, month, day by looking for the NATIVE directory
        try:
            native_index = path_parts.index("NATIVE")
            year = path_parts[native_index + 1]
            month = path_parts[native_index + 2]
            # The next part could be either a day or a filename, so we need to check
            if len(path_parts) > native_index + 3:
                day_part = path_parts[native_index + 3]
                # If it's a day directory (should be two digits)
                if day_part.isdigit() and len(day_part) == 2:
                    day = day_part
                else:
                    day = "01"  # Default if we can't determine
            else:
                day = "01"  # Default if we can't determine
        except (ValueError, IndexError):
            # If we can't parse the path, use default values
            year = "2025"
            month = "01"
            day = "01"
            print(f"Warning: Could not parse date from path: {gz_file}")
        
        with gzip.open(gz_file, 'rt') as f:
            # Read and filter immediately to save memory
            df = pd.read_csv(f, low_memory=False)
            
            # Apply bounding box filter
            df_filtered = filter_by_bbox(df, bbox_coords)
            
            if not df_filtered.empty:
                # Use .loc to avoid SettingWithCopyWarning
                df_filtered = df_filtered.copy()
                df_filtered.loc[:, 'source_file'] = os.path.basename(gz_file)
                df_filtered.loc[:, 'acquisition_date'] = f"{year}-{month}-{day}"
                
                # Convert ACQTIME to proper datetime
                if 'ACQTIME' in df_filtered.columns:
                    df_filtered.loc[:, 'acquisition_datetime'] = df_filtered['ACQTIME'].apply(parse_acqtime)
                else:
                    print(f"Warning: ACQTIME column not found in {gz_file}")
                
                return df_filtered
            
    except Exception as e:
        print(f"Error processing {gz_file}: {str(e)}")
    
    return None

def decompress_and_aggregate(base_dir, year, month=None, day=None, bbox_coords=None, output_filename=None):
    """
    Decompress all .gz files with bounding box filtering for the specified period
    """
    if bbox_coords is None:
        bbox_coords = (36.87164804628416, -9.633111264309846, 42.24431922230131, -6.070242597727865)
    
    # Determine the search pattern based on the period
    if day and month:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, month, day, "*.gz")
    elif month:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, month, "*", "*.gz")
    else:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, "*", "*", "*.gz")

    gz_files = glob.glob(search_pattern)
    
    # If no files found, try recursive search as fallback
    if not gz_files:
        alt_pattern = os.path.join(base_dir, "**", "*.gz")
        gz_files = glob.glob(alt_pattern, recursive=True)
        # Filter for the correct period
        if day and month:
            gz_files = [f for f in gz_files if f"/{year}/{month}/{day}/" in f]
        elif month:
            gz_files = [f for f in gz_files if f"/{year}/{month}/" in f]
        else:
            gz_files = [f for f in gz_files if f"/{year}/" in f]

    if not gz_files:
        print(f"No .gz files found for the specified period.")
        return None
    
    print(f"Found {len(gz_files)} compressed files to process...")
    lat_min, lon_min, lat_max, lon_max = bbox_coords
    print(f"Applying bounding box filter: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°W to {lon_max:.2f}°W")
    
    all_dataframes = []
    processed_count = 0
    total_records = 0
    
    for i, gz_file in enumerate(gz_files):
        df_filtered = process_single_file(gz_file, bbox_coords)
        
        if df_filtered is not None:
            all_dataframes.append(df_filtered)
            processed_count += 1
            total_records += len(df_filtered)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(gz_files)} files... Found {len(df_filtered)} records in this file. Total so far: {total_records}")
        
        # Clear memory periodically
        if (i + 1) % 50 == 0:
            import gc
            gc.collect()
    
    if not all_dataframes:
        print("No files were successfully processed.")
        return None
    
    print(f"Combining {len(all_dataframes)} filtered datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Saving combined data to {output_filename}...")
    combined_df.to_csv(output_filename, index=False)
    
    print(f"\nAggregation complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Total records after filtering: {len(combined_df)}")
    
    # Show datetime range if available
    if 'acquisition_datetime' in combined_df.columns:
        min_dt = combined_df['acquisition_datetime'].min()
        max_dt = combined_df['acquisition_datetime'].max()
        print(f"Datetime range: {min_dt} to {max_dt}")
    
    return output_filename

def create_qgis_ready_geopackage(csv_filename, lat_col='LATITUDE', lon_col='LONGITUDE'):
    """
    Convert filtered CSV to GeoPackage for QGIS with proper datetime field
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        print(f"Reading filtered CSV: {csv_filename}")
        df = pd.read_csv(csv_filename)
        
        # Convert acquisition_datetime string back to datetime object
        if 'acquisition_datetime' in df.columns:
            df['acquisition_datetime'] = pd.to_datetime(df['acquisition_datetime'])
        
        print("Creating geometries...")
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        gpkg_filename = csv_filename.replace('.csv', '.gpkg')
        gdf.to_file(gpkg_filename, driver="GPKG")
        
        print(f"GeoPackage created: {gpkg_filename}")
        
        # Print some stats about the data
        if 'acquisition_datetime' in gdf.columns:
            print(f"Datetime range in GeoPackage: {gdf['acquisition_datetime'].min()} to {gdf['acquisition_datetime'].max()}")
        
        return gpkg_filename
        
    except ImportError:
        print("geopandas not available. Install with: pip install geopandas")
        return csv_filename

def process_in_batches(base_dir, year, month=None, day=None, bbox_coords=None, batch_size=10, output_filename=None):
    """
    Alternative method: Process files in batches to save memory
    """
    if bbox_coords is None:
        bbox_coords = (36.87164804628416, -9.633111264309846, 42.24431922230131, -6.070242597727865)
    
    # Determine the search pattern based on the period
    if day and month:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, month, day, "*.gz")
    elif month:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, month, "*", "*.gz")
    else:
        search_pattern = os.path.join(base_dir, "PRODUCTS", "MTG", "MTFRPPixel", "NATIVE", year, "*", "*", "*.gz")

    gz_files = glob.glob(search_pattern)
    
    # If no files found, try recursive search as fallback
    if not gz_files:
        alt_pattern = os.path.join(base_dir, "**", "*.gz")
        gz_files = glob.glob(alt_pattern, recursive=True)
        # Filter for the correct period
        if day and month:
            gz_files = [f for f in gz_files if f"/{year}/{month}/{day}/" in f]
        elif month:
            gz_files = [f for f in gz_files if f"/{year}/{month}/" in f]
        else:
            gz_files = [f for f in gz_files if f"/{year}/" in f]

    if not gz_files:
        return None
    
    # Create temp directory name based on period
    temp_dir = f"temp_{year}"
    if month:
        temp_dir += f"_{month}"
    if day:
        temp_dir += f"_{day}"
    Path(temp_dir).mkdir(exist_ok=True)
    
    print(f"Processing {len(gz_files)} files in batches of {batch_size}...")
    
    # Process files in batches and save temporary CSVs
    for batch_num, i in enumerate(range(0, len(gz_files), batch_size)):
        batch_files = gz_files[i:i + batch_size]
        batch_dfs = []
        
        print(f"Processing batch {batch_num + 1}...")
        
        for gz_file in batch_files:
            df = process_single_file(gz_file, bbox_coords)
            if df is not None:
                batch_dfs.append(df)
        
        if batch_dfs:
            # Save this batch to a temporary file
            batch_combined = pd.concat(batch_dfs, ignore_index=True)
            temp_filename = os.path.join(temp_dir, f"batch_{batch_num}.csv")
            batch_combined.to_csv(temp_filename, index=False)
            print(f"Batch {batch_num + 1}: {len(batch_combined)} records saved to temporary file")
        
        # Clear memory
        import gc
        gc.collect()
    
    # Combine all temporary files
    temp_files = glob.glob(os.path.join(temp_dir, "batch_*.csv"))
    
    if not temp_files:
        print("No data processed.")
        return None
    
    print(f"Combining {len(temp_files)} temporary files...")
    
    # Read and combine all temporary files
    all_dataframes = []
    for temp_file in temp_files:
        df = pd.read_csv(temp_file)
        all_dataframes.append(df)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(output_filename, index=False)
    
    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    os.rmdir(temp_dir)
    
    return output_filename

def main():
    """
    Main function with CLI arguments
    """
    parser = argparse.ArgumentParser(description='Download and process MTG/MTFRPPixel data from LSA SAF')
    
    # Required arguments
    parser.add_argument('--username', required=True, help='LSA SAF username')
    parser.add_argument('--password', required=True, help='LSA SAF password')
    parser.add_argument('--year', required=True, help='Year in YYYY format')
    
    # Optional arguments
    parser.add_argument('--month', required=False, help='Month in MM format')
    parser.add_argument('--day', required=False, help='Day in DD format')
    parser.add_argument('--base_dir', default='FRP_MTG', help='Base directory for downloads (default: FRP_MTG)')
    parser.add_argument('--bbox', default='36.87164804628416,-9.633111264309846,42.24431922230131,-6.070242597727865',
                       help='Bounding box: lat_min,lon_min,lat_max,lon_max (default: Portugal bbox)')
    parser.add_argument('--output_name', required=False, help='Base name for output files (without extension)')
    parser.add_argument('--skip_download', action='store_true', 
                       help='Skip download and process existing data in base_dir')
    parser.add_argument('--skip_geopackage', action='store_true',
                       help='Skip GeoPackage conversion, only create CSV')
    parser.add_argument('--in_memory', action='store_true',
                       help='Download directly to memory without saving files to disk')
    
    args = parser.parse_args()
    
    try:
        year_int = int(args.year)
        if year_int < 2025:
            print(f"Error: MTG/MTFRPPixel data is only available from 2025 onwards. You specified {args.year}.")
            print("Please use a year >= 2025.")
            sys.exit(1)
    except ValueError:
        print(f"Error: Invalid year format '{args.year}'. Please use YYYY format.")
        sys.exit(1)
    
    if args.month:
        try:
            month_int = int(args.month)
            if not (1 <= month_int <= 12):
                print(f"Error: Month must be between 1 and 12. You specified {args.month}.")
                sys.exit(1)
            args.month = f"{month_int:02d}"
        except ValueError:
            print(f"Error: Invalid month format '{args.month}'. Please use numeric format (1-12).")
            sys.exit(1)
    
    if args.day:
        try:
            day_int = int(args.day)
            if not (1 <= day_int <= 31):
                print(f"Error: Day must be between 1 and 31. You specified {args.day}.")
                sys.exit(1)
            args.day = f"{day_int:02d}"
        except ValueError:
            print(f"Error: Invalid day format '{args.day}'. Please use numeric format (1-31).")
            sys.exit(1)
    
    # Parse bounding box
    bbox_coords = parse_bbox(args.bbox)
    
    # Determine output filename
    if args.output_name:
        base_output = args.output_name
    else:
        base_output = "MTG_MTFRPPixel_aggregated"
        
    if args.day and args.month:
        output_filename = f"{base_output}_{args.year}_{args.month}_{args.day}.csv"
    elif args.month:
        output_filename = f"{base_output}_{args.year}_{args.month}.csv"
    else:
        output_filename = f"{base_output}_{args.year}.csv"
    
    print("MTG/MTFRPPixel Data Processor")
    print("=" * 50)
    period_str = f"{args.year}"
    if args.month:
        period_str += f"-{args.month}"
    if args.day:
        period_str += f"-{args.day}"
    print(f"Period: {period_str}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output file: {output_filename}")
    print(f"Bounding box: {bbox_coords}")
    print(f"In memory processing: {args.in_memory}")
    print("=" * 50)
    
    # Step 1: Download data (unless skipped or using in_memory)
    download_needed = True
    
    if args.in_memory:
        print("Using in-memory processing (no files saved to disk)...")
        download_needed = False
    elif args.skip_download:
        print("Skipping download as requested...")
        download_needed = False
    elif check_existing_download(args.base_dir, args.year, args.month, args.day):
        print(f"Data already exists in {args.base_dir}. Skipping download.")
        download_needed = False
        print("If you want to re-download, use a different base_dir or delete the existing directory.")
    
    if download_needed:
        print("Starting download using requests...")
        success = download_data_requests(args.year, args.month, args.day, args.username, args.password, args.base_dir)
        
        if not success:
            print("Download failed. Please check credentials and try again.")
            return
    elif not args.in_memory:
        print("Using existing data for processing...")
    
    # Choose processing method
    if args.in_memory:
        print("Using in-memory processing...")
        output_csv = process_in_memory(args.year, args.month, args.day, args.username, 
                                      args.password, bbox_coords, output_filename)
    else:
        if sys.stdout.isatty():  # Only ask if running in terminal
            print("\nChoose processing method:")
            print("1. Standard processing (faster, uses more RAM)")
            print("2. Batch processing (slower, uses less RAM)")
            choice = input("Enter choice (1 or 2, default 1): ").strip() or "1"
        else:
            choice = "1"  # Default to standard processing in non-interactive mode
        
        if choice == "2":
            print("Using batch processing to save RAM...")
            output_csv = process_in_batches(args.base_dir, args.year, args.month, args.day, bbox_coords, 
                                           batch_size=10, output_filename=output_filename)
        else:
            print("Using standard processing...")
            output_csv = decompress_and_aggregate(args.base_dir, args.year, args.month, args.day, 
                                                 bbox_coords, output_filename)
    
    if output_csv:
        # Check file size
        file_size = os.path.getsize(output_csv) / (1024 * 1024)  # MB
        print(f"Output file size: {file_size:.2f} MB")
        
        # Convert to GeoPackage unless skipped
        if not args.skip_geopackage:
            if sys.stdout.isatty() and not args.skip_geopackage:
                convert = input("Convert to GeoPackage for QGIS? (y/n, default y): ").strip().lower() or "y"
            else:
                convert = "y"  # Default to yes in non-interactive mode
                
            if convert == "y":
                final_file = create_qgis_ready_geopackage(output_csv)
                print(f"QGIS-ready file: {final_file}")
            else:
                final_file = output_csv
        else:
            final_file = output_csv
            print(f"CSV file: {final_file} (GeoPackage conversion skipped)")
            
        print(f"\nProcessing complete! You can now open {final_file} in QGIS.")
        
        # Show usage examples
        print("\nUsage examples for next time:")
        print(f"  python {sys.argv[0]} --username {args.username} --password {args.password} --year {args.year}")
        print(f"  python {sys.argv[0]} --username {args.username} --password {args.password} --year {args.year} --month {args.month}")
        print(f"  python {sys.argv[0]} --username {args.username} --password {args.password} --year {args.year} --month {args.month} --day {args.day}")
        print(f"  python {sys.argv[0]} --username {args.username} --password {args.password} --year {args.year} --output_name my_custom_name")
        print(f"  python {sys.argv[0]} --username {args.username} --password {args.password} --year {args.year} --in_memory")

if __name__ == "__main__":
    main()