#!/usr/bin/env python3
import os
import json
import csv
import io
import math
import argparse
import asyncio
import aiofiles

async def process_file(input_path: str, output_folder: str, limit_bytes: int):
    """
    Process a single JSONL file:
      - Read the file asynchronously line by line.
      - For each valid JSON line, extract selected fields.
      - Write CSV rows to an output file.
      - If the current CSV file exceeds the size limit, start a new CSV file.
    """
    csv_files = []
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    file_counter = 1
    output_filename = os.path.join(output_folder, f"{base_name}_{file_counter}.csv")
    out_f = await aiofiles.open(output_filename, mode="w", encoding="utf-8")
    current_size = 0

    # CSV header and pre-generated header string
    headers = ["rating", "title", "text"]
    header_io = io.StringIO()
    header_writer = csv.DictWriter(header_io, fieldnames=headers)
    header_writer.writeheader()
    header_str = header_io.getvalue()
    header_bytes = header_str.encode("utf-8")
    header_len = len(header_bytes)

    # Write header into the first CSV file.
    await out_f.write(header_str)
    current_size += header_len

    async with aiofiles.open(input_path, mode="r", encoding="utf-8") as in_f:
        async for line in in_f:
            line = line.strip()
            if not line:
                continue

            try:
                json_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {input_path}")
                continue

            # Extract selected columns.
            filtered_data = {
                "rating": json_data.get("rating"),
                "title": json_data.get("title"),
                "text": json_data.get("text")
            }

            # Create CSV row as a string.
            row_io = io.StringIO()
            row_writer = csv.DictWriter(row_io, fieldnames=headers)
            row_writer.writerow(filtered_data)
            row_str = row_io.getvalue()
            row_bytes = row_str.encode("utf-8")
            row_len = len(row_bytes)

            # If adding this row would exceed the limit, close the current file and start a new one.
            if current_size + row_len > limit_bytes:
                await out_f.close()
                csv_files.append(output_filename)
                print(f"Finished {output_filename} (size: {current_size} bytes)")
                file_counter += 1
                output_filename = os.path.join(output_folder, f"{base_name}_{file_counter}.csv")
                out_f = await aiofiles.open(output_filename, mode="w", encoding="utf-8")
                # Write header for the new CSV file.
                await out_f.write(header_str)
                current_size = header_len

            await out_f.write(row_str)
            current_size += row_len

    await out_f.close()
    csv_files.append(output_filename)
    print(f"Finished processing {input_path}. Created {file_counter} CSV file(s) for this input.")
    return csv_files

async def split_csv_file_async(file_path: str, parts: int):
    """
    Split the given CSV file into a specified number of smaller CSV files.
    The output files will have names like <original_basename>_part1.csv, etc.
    The header line is preserved in every output file.
    """
    base_name, _ = os.path.splitext(file_path)
    # First, count total rows (excluding header).
    total_rows = 0
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
        header = await f.readline()  # read header
        async for _ in f:
            total_rows += 1
    if total_rows == 0:
        print(f"No rows to split in {file_path}")
        return

    rows_per_part = math.ceil(total_rows / parts)
    print(f"Splitting {file_path} into {parts} parts (~{rows_per_part} rows per part).")

    # Open part files for writing.
    part_files = []
    for i in range(parts):
        part_filename = f"{base_name}_part{i+1}.csv"
        pf = await aiofiles.open(part_filename, mode="w", encoding="utf-8")
        part_files.append(pf)

    # Write header into each part.
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
        header_line = await f.readline()  # header line
    for pf in part_files:
        await pf.write(header_line)

    # Distribute rows across part files.
    current_row = 0
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
        await f.readline()  # skip header
        async for line in f:
            part_index = current_row // rows_per_part
            if part_index >= parts:
                part_index = parts - 1
            await part_files[part_index].write(line)
            current_row += 1

    for pf in part_files:
        await pf.close()

    print(f"Finished splitting {file_path} into {parts} parts.")

async def main():
    parser = argparse.ArgumentParser(
        description="Asynchronously convert JSONL files to CSV files (with selected columns) while splitting "
                    "output files to keep each under a specified size (in GB)."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="resources",
        help="Folder containing JSONL files (default: 'resources')"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="csv_output",
        help="Folder to save CSV output files (default: 'csv_output')"
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=10.0,
        help="Maximum CSV file size in GB (default: 10.0)"
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=1,
        help="Number of smaller files to split each CSV chunk into (default: 1, i.e. no further splitting)"
    )
    args = parser.parse_args()

    # Convert GB limit to bytes.
    limit_bytes = int(args.limit * (1024 ** 3))
    print(f"CSV file size limit: {limit_bytes} bytes (~{args.limit} GB)")

    os.makedirs(args.output_folder, exist_ok=True)
    # List JSONL files in the input folder.
    jsonl_files = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder) if f.endswith(".jsonl")
    ]
    if not jsonl_files:
        print(f"No JSONL files found in '{args.input_folder}'.")
        return

    # Create asynchronous tasks to process each file concurrently.
    tasks = [process_file(path, args.output_folder, limit_bytes) for path in jsonl_files]
    results = await asyncio.gather(*tasks)

    # Flatten the list of CSV files.
    all_csv_files = [csv_file for sublist in results for csv_file in sublist]

    # If splitting into parts is requested, split each produced CSV file.
    if args.parts > 1:
        split_tasks = [split_csv_file_async(csv_file, args.parts) for csv_file in all_csv_files]
        await asyncio.gather(*split_tasks)

    print("\n✅ All files have been processed and saved in the output folder.")

if __name__ == '__main__':
    asyncio.run(main())


# To run this script:
####
# python async_jsonl_to_csv_split.py --input_folder resources --output_folder csv_output --limit 10 --parts 3

