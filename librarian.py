import anthropic
from anthropic import AsyncAnthropic
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import aiohttp
import xml.etree.ElementTree as ET
import base64
import duckdb
from functools import wraps
from cachetools import TTLCache, cached
import hashlib
import signal
import math
import fitz  # PyMuPDF
from PIL import Image
import io
from datetime import datetime, timezone
import tiktoken
import json
import time
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT = 1  # Requests per second
MAX_CHUNK_SIZE = 50000  # Adjust this value based on testing
MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds

# Create TTL cache for papers and summaries
paper_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache papers for 1 hour
summary_cache = TTLCache(maxsize=5000, ttl=86400)  # Cache summaries for 1 day

class ArxivAPI:
    def __init__(self):
        self.paper_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache papers for 1 hour

    async def search_papers(self, query: str, max_results: int = 2) -> List[Dict]:
        cache_key = f"{query}_{max_results}"
        if cache_key in self.paper_cache:
            return self.paper_cache[cache_key]

        logger.info(f"Searching arXiv for query: '{query}' (max results: {max_results})")
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ARXIV_BASE_URL, params=params) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        root = ET.fromstring(response_text)
                        papers = []
                        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                            paper = {
                                "id": entry.find('{http://www.w3.org/2005/Atom}id').text,
                                "title": entry.find('{http://www.w3.org/2005/Atom}title').text,
                                "abstract": entry.find('{http://www.w3.org/2005/Atom}summary').text,
                                "pdf_url": entry.find("{http://www.w3.org/2005/Atom}link[@title='pdf']").get('href')
                            }
                            papers.append(paper)
                        logger.info(f"Found {len(papers)} papers for query: '{query}'")
                        self.paper_cache[cache_key] = papers
                        return papers
                    else:
                        logger.error(f"arXiv API error: {response.status}")
                        return []
        except aiohttp.ClientError as e:
            logger.error(f"Network error when querying arXiv: {str(e)}")
            return []

class ResearchDatabase:
    def __init__(self, db_path: str = 'research_papers_v15.db'):
        self.db_path = db_path
        self.conn = None
        self.tag_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache tags for 1 hour

    def connect(self):
        self.conn = duckdb.connect(self.db_path)
        self.create_tables()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def commit(self):
        logger.info("Committing changes to the database")
        if self.conn:
            self.conn.commit()
        else:
            logger.warning("No active connection to commit changes")

    def rollback(self):
        logger.info("Rolling back changes in the database")
        if self.conn:
            self.conn.rollback()
        else:
            logger.warning("No active connection to rollback changes")

    def create_tables(self):
        logger.info("Creating database tables if they don't exist")
        
        # Create sequences for auto-incrementing IDs
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS tag_summary_seq")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS technical_detail_seq")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS key_finding_seq")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS methodology_seq")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS novel_contribution_seq")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS future_direction_seq")
        
        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id VARCHAR PRIMARY KEY,
                title VARCHAR,
                abstract VARCHAR,
                pdf_url VARCHAR,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_summaries (
                summary_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('tag_summary_seq'),
                paper_id VARCHAR,
                primary_tag VARCHAR,
                secondary_tag VARCHAR,
                analysis TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_details (
                detail_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('technical_detail_seq'),
                summary_id INTEGER,
                detail_type VARCHAR,
                content TEXT,
                explanation TEXT,
                dependencies TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (summary_id) REFERENCES tag_summaries (summary_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS key_findings (
                finding_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('key_finding_seq'),
                summary_id INTEGER,
                finding TEXT,
                FOREIGN KEY (summary_id) REFERENCES tag_summaries (summary_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS methodologies (
                methodology_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('methodology_seq'),
                summary_id INTEGER,
                methodology TEXT,
                FOREIGN KEY (summary_id) REFERENCES tag_summaries (summary_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS novel_contributions (
                contribution_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('novel_contribution_seq'),
                summary_id INTEGER,
                contribution TEXT,
                FOREIGN KEY (summary_id) REFERENCES tag_summaries (summary_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS future_directions (
                direction_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('future_direction_seq'),
                summary_id INTEGER,
                direction TEXT,
                FOREIGN KEY (summary_id) REFERENCES tag_summaries (summary_id)
            )
        """)
        logger.info("Database tables created successfully")

    def insert_key_finding(self, summary_id: int, finding: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting key finding for summary {summary_id}")
            result = self.conn.execute("""
                INSERT INTO key_findings (summary_id, finding)
                VALUES (?, ?)
                RETURNING finding_id
            """, (summary_id, finding)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def insert_methodology(self, summary_id: int, methodology: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting methodology for summary {summary_id}")
            result = self.conn.execute("""
                INSERT INTO methodologies (summary_id, methodology)
                VALUES (?, ?)
                RETURNING methodology_id
            """, (summary_id, methodology)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def insert_novel_contribution(self, summary_id: int, contribution: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting novel contribution for summary {summary_id}")
            result = self.conn.execute("""
                INSERT INTO novel_contributions (summary_id, contribution)
                VALUES (?, ?)
                RETURNING contribution_id
            """, (summary_id, contribution)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def insert_future_direction(self, summary_id: int, direction: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting future direction for summary {summary_id}")
            result = self.conn.execute("""
                INSERT INTO future_directions (summary_id, direction)
                VALUES (?, ?)
                RETURNING direction_id
            """, (summary_id, direction)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def insert_paper(self, paper: Dict):
        self.connect()
        try:
            logger.info(f"Inserting/updating paper in database: {paper['id']}")
            self.conn.execute("""
                INSERT OR REPLACE INTO papers (paper_id, title, abstract, pdf_url, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (paper['id'], paper['title'], paper['abstract'], paper['pdf_url']))
        finally:
            self.close()

    def insert_tag_summary(self, paper_id: str, primary_tag: str, secondary_tag: str, analysis: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting tag summary for paper {paper_id}: {primary_tag} - {secondary_tag}")
            result = self.conn.execute("""
                INSERT INTO tag_summaries (paper_id, primary_tag, secondary_tag, analysis, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING summary_id
            """, (paper_id, primary_tag, secondary_tag, analysis)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def insert_technical_detail(self, summary_id: int, detail_type: str, content: str, explanation: str, dependencies: str) -> int:
        self.connect()
        try:
            logger.info(f"Inserting technical detail for summary {summary_id}: {detail_type}")
            result = self.conn.execute("""
                INSERT INTO technical_details (summary_id, detail_type, content, explanation, dependencies, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING detail_id
            """, (summary_id, detail_type, content, explanation, dependencies)).fetchone()
            return result[0] if result else None
        finally:
            self.close()

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        self.connect()
        try:
            logger.info(f"Fetching paper from database: {paper_id}")
            result = self.conn.execute("""
                SELECT paper_id, title, abstract, pdf_url, last_updated
                FROM papers WHERE paper_id = ?
            """, (paper_id,)).fetchone()
            
            if result:
                return {
                    'id': result[0],
                    'title': result[1],
                    'abstract': result[2],
                    'pdf_url': result[3],
                    'last_updated': result[4]
                }
            return None
        finally:
            self.close()

    def get_tag_summaries(self, paper_id: str) -> List[Dict]:
        self.connect()
        try:
            logger.info(f"Fetching tag summaries for paper: {paper_id}")
            results = self.conn.execute("""
                SELECT summary_id, paper_id, primary_tag, secondary_tag, analysis, last_updated
                FROM tag_summaries WHERE paper_id = ?
            """, (paper_id,)).fetchall()
            return [
                {
                    'summary_id': row[0],
                    'paper_id': row[1],
                    'primary_tag': row[2],
                    'secondary_tag': row[3],
                    'analysis': row[4],
                    'last_updated': row[5]
                }
                for row in results
            ]
        finally:
            self.close()

    def get_technical_details(self, summary_id: int) -> List[Dict]:
        self.connect()
        try:
            logger.info(f"Fetching technical details for summary: {summary_id}")
            results = self.conn.execute("""
                SELECT detail_id, summary_id, detail_type, content, dependencies, last_updated
                FROM technical_details WHERE summary_id = ?
            """, (summary_id,)).fetchall()
            return [
                {
                    'detail_id': row[0],
                    'summary_id': row[1],
                    'detail_type': row[2],
                    'content': row[3],
                    'dependencies': row[4],
                    'last_updated': row[5]
                }
                for row in results
            ]
        finally:
            self.close()

    def get_all_tags(self) -> List[Tuple[str, str]]:
        if self.tag_cache:
            return list(self.tag_cache.values())

        self.connect()
        try:
            logger.info("Fetching all tags from database")
            results = self.conn.execute("""
                SELECT DISTINCT primary_tag, secondary_tag
                FROM tag_summaries
            """).fetchall()
            tags = [(row[0], row[1]) for row in results]
            self.tag_cache = TTLCache(maxsize=1000, ttl=3600)
            for i, tag in enumerate(tags):
                self.tag_cache[i] = tag
            return tags
        finally:
            self.close()


class Orchestrator:
    def __init__(self, api_key: str):
        logger.info("Initializing Orchestrator")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.arxiv_api = ArxivAPI()
        self.db = ResearchDatabase()
        self.db.connect()  # Open a persistent connection
        self.api_semaphore = asyncio.Semaphore(1)  # Allow only 1 API call at a time
        self.chunk_cache = TTLCache(maxsize=500, ttl=3600)  # Cache for 1 hour
        logger.info("Orchestrator initialized successfully")

    def close_db(self):
        if hasattr(self, 'db'):
            logger.info("Closing database connection")
            self.db.close()
            logger.info("Database connection closed")

    async def execute_system(self, initial_prompt: str):
        try:
            logger.info(f"Executing system with initial prompt: '{initial_prompt}'")
            keywords = await self.generate_keywords(initial_prompt)
            if not keywords:
                logger.warning("No keywords generated. Exiting system execution.")
                return
            logger.info(f"Generated keywords: {keywords}")
            for keyword in keywords:
                logger.info(f"Processing keyword: '{keyword}'")
                papers = await self.arxiv_api.search_papers(keyword)
                if not papers:
                    logger.warning(f"No papers found for keyword: '{keyword}'")
                    continue
                for paper in papers:
                    logger.info(f"Processing paper: {paper['id']}")
                    try:
                        await self.process_paper(paper)
                        self.db.commit()  # Commit after each paper is fully processed
                        logger.info(f"Committed changes for paper: {paper['id']}")
                    except Exception as e:
                        logger.error(f"Error processing paper {paper['id']}: {str(e)}")
                        self.db.rollback()  # Rollback changes if an error occurs during paper processing
            logger.info("System execution completed")
        except Exception as e:
            logger.error(f"An error occurred during system execution: {str(e)}")
            self.db.rollback()  # Rollback changes if an error occurs
        finally:
            self.close_db()

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    async def generate_keywords(self, prompt: str) -> List[str]:
        logger.info(f"Generating keywords for prompt: '{prompt}'")
        async with self.api_semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempt {attempt + 1} to generate keywords")
                    response = await self.client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=8000,
                        messages=[
                            {"role": "user", "content": f"Avoid preambles and posteables, just generate a list of 5. Generate 5 key search terms for the following topic: {prompt}"}
                        ]
                    )
                    keywords = response.content[0].text.strip().split('\n')
                    # Remove any numbering or extra spaces
                    keywords = [k.split('.', 1)[-1].strip() for k in keywords]
                    logger.info(f"Keywords generated successfully: {keywords}")
                    return keywords
                except Exception as e:
                    if "Number of request tokens" in str(e):
                        logger.warning(f"Token limit reached. Closing DB connection and waiting for 60 seconds before retrying.")
                        self.db.close()
                        await asyncio.sleep(60)
                        self.db = ResearchDatabase()  # Reopen the database connection
                    elif attempt < MAX_RETRIES - 1:
                        logger.warning(f"API call failed, retrying in {RETRY_DELAY} seconds. Error: {str(e)}")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"API call failed after {MAX_RETRIES} attempts. Error: {str(e)}")
                        raise
        return []  # Return an empty list if all attempts fail

    async def process_paper(self, paper: Dict[str, Any]):
        logger.info(f"Processing paper: {paper['id']}")
        cached_paper = self.db.get_paper(paper['id'])
        if cached_paper:
            logger.info(f"Paper {paper['id']} found in cache")
            cached_summaries = self.db.get_tag_summaries(paper['id'])
            if cached_summaries:
                logger.info(f"Tag summaries for paper {paper['id']} found in cache")
                return

        self.db.insert_paper(paper)
        
        if paper['pdf_url'].startswith('http'):
            logger.info(f"Downloading PDF for paper: {paper['id']}")
            pdf_content = await self.download_pdf(paper['pdf_url'])
        else:
            logger.info(f"Reading local PDF for paper: {paper['id']}")
            with open(paper['pdf_url'], 'rb') as file:
                pdf_content = file.read()
        
        logger.info(f"Splitting PDF content for paper: {paper['id']}")
        chunks = self.split_pdf(pdf_content)
        logger.info(f"Analyzing {len(chunks)} chunks for paper: {paper['id']}")
        
        for i, chunk in enumerate(chunks, 1):
            try:
                result = await self.analyze_chunk(paper['id'], paper['title'], paper['abstract'], chunk, i, len(chunks))
                if result:
                    for primary_tag, secondary_tags in result.items():
                        if not isinstance(secondary_tags, dict):
                            logger.warning(f"Unexpected format for secondary tags in chunk {i}: {secondary_tags}")
                            continue
                        for secondary_tag, analysis in secondary_tags.items():
                            if not isinstance(analysis, dict):
                                logger.warning(f"Unexpected format for analysis in chunk {i}: {analysis}")
                                continue
                            
                            analysis_text = analysis.get('analysis', '')
                            if analysis_text:
                                summary_id = self.db.insert_tag_summary(paper['id'], primary_tag, secondary_tag, f"Chunk {i}/{len(chunks)}: {analysis_text}")
                                if summary_id is None:
                                    logger.error(f"Failed to insert tag summary for chunk {i}/{len(chunks)} of paper {paper['id']}")
                                    continue
                                
                                # Process technical details
                                technical_details = analysis.get('technical_details', [])
                                for detail in technical_details:
                                    if isinstance(detail, dict) and all(k in detail for k in ('type', 'content', 'explanation', 'dependencies')):
                                        detail_id = self.db.insert_technical_detail(
                                            summary_id,
                                            detail['type'],
                                            detail['content'],
                                            detail['explanation'],
                                            detail['dependencies']
                                        )
                                        if detail_id is None:
                                            logger.warning(f"Failed to insert technical detail for summary {summary_id}")
                                    else:
                                        logger.warning(f"Invalid technical detail format: {detail}")
                                
                                # Process key findings
                                key_findings = analysis.get('key_findings', [])
                                for finding in key_findings:
                                    finding_id = self.db.insert_key_finding(summary_id, finding)
                                    if finding_id is None:
                                        logger.warning(f"Failed to insert key finding for summary {summary_id}")
                                
                                # Process methodologies
                                methodologies = analysis.get('methodologies', [])
                                for methodology in methodologies:
                                    methodology_id = self.db.insert_methodology(summary_id, methodology)
                                    if methodology_id is None:
                                        logger.warning(f"Failed to insert methodology for summary {summary_id}")
                                
                                # Process novel contributions
                                novel_contributions = analysis.get('novel_contributions', [])
                                for contribution in novel_contributions:
                                    contribution_id = self.db.insert_novel_contribution(summary_id, contribution)
                                    if contribution_id is None:
                                        logger.warning(f"Failed to insert novel contribution for summary {summary_id}")
                                
                                # Process future directions
                                future_directions = analysis.get('future_directions', [])
                                for direction in future_directions:
                                    direction_id = self.db.insert_future_direction(summary_id, direction)
                                    if direction_id is None:
                                        logger.warning(f"Failed to insert future direction for summary {summary_id}")
                                
                                logger.info(f"Processed chunk {i}/{len(chunks)} for paper {paper['id']}")
                            else:
                                logger.warning(f"Empty analysis for chunk {i}/{len(chunks)} of paper {paper['id']}")
            except Exception as e:
                logger.error(f"Error processing chunk {i} for paper {paper['id']}: {str(e)}")
        
        logger.info(f"Completed processing paper: {paper['id']}")

    @cached(cache=TTLCache(maxsize=500, ttl=86400))
    async def download_pdf(self, pdf_url: str) -> bytes:
        logger.info(f"Downloading PDF from URL: {pdf_url}")
        async with self.api_semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempt {attempt + 1} to download PDF")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(pdf_url) as response:
                            response.raise_for_status()
                            content = await response.read()
                            logger.info(f"PDF downloaded successfully, size: {len(content)} bytes")
                            return content
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"PDF download failed, retrying in {RETRY_DELAY} seconds. Error: {str(e)}")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"PDF download failed after {MAX_RETRIES} attempts. Error: {str(e)}")
                        raise

    def split_pdf(self, pdf_content: bytes, target_chunk_size: int = 20000) -> List[Dict[str, Any]]:
        logger.info(f"Splitting PDF content, total size: {len(pdf_content)} bytes")
        
        # If pdf_content is a file path, read the file
        if isinstance(pdf_content, str):
            with open(pdf_content, 'rb') as file:
                pdf_content = file.read()
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        chunks = []
        current_chunk = {"text": "", "images": []}
        current_chunk_size = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            images = self.extract_images(page)
            
            # If adding this page would exceed the target chunk size, start a new chunk
            if current_chunk_size + len(text) > target_chunk_size and current_chunk["text"]:
                chunks.append(current_chunk)
                current_chunk = {"text": "", "images": []}
                current_chunk_size = 0

            current_chunk["text"] += text
            current_chunk["images"].extend(images)
            current_chunk_size += len(text)

        # Add the last chunk if it's not empty
        if current_chunk["text"] or current_chunk["images"]:
            chunks.append(current_chunk)

        logger.info(f"PDF split into {len(chunks)} chunks")
        return chunks
    
    def extract_images(self, page: fitz.Page) -> List[Dict[str, str]]:
        images = []
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert image to base64
            image = Image.open(io.BytesIO(image_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            images.append({
                "data": img_str,
                "type": base_image["ext"],
                "size": f"{image.width}x{image.height}"
            })
        return images
    
    async def analyze_chunk(self, paper_id: str, title: str, abstract: str, chunk: Dict[str, Any], chunk_number: int, total_chunks: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
        cache_key = self._get_cache_key(paper_id, chunk["text"])
        if cache_key in self.chunk_cache:
            logger.info(f"Analysis for chunk {chunk_number}/{total_chunks} of paper {paper_id} found in cache")
            return self.chunk_cache[cache_key]

        logger.info(f"Analyzing chunk {chunk_number}/{total_chunks} for paper: {paper_id}")
        async with self.api_semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempt {attempt + 1} to analyze chunk {chunk_number}/{total_chunks}")
                    existing_tags = self.db.get_all_tags()
                    
                    image_descriptions = [f"Image {i+1}: {img['size']} {img['type']} image" for i, img in enumerate(chunk["images"])]
                    
                    prompt = f"""
                    Analyze the following chunk ({chunk_number}/{total_chunks}) of a research paper:
                    Title: {title}
                    Abstract: {abstract}

                    Text content:
                    {chunk["text"]}

                    Images in this chunk:
                    {', '.join(image_descriptions)}

                    Provide an extremely detailed analysis of this chunk, extracting as much information as possible. Your analysis should include:

                    1. Key concepts and ideas presented
                    2. Methodologies described or used
                    3. Results and findings
                    4. Conclusions or implications
                    5. Any hypotheses or theories proposed
                    6. Experimental setups or data collection methods
                    7. Statistical analyses or mathematical models used
                    8. Limitations or constraints mentioned
                    9. Future research directions suggested
                    10. Connections to other works or fields
                    11. Any novel contributions or innovations

                    Generate appropriate tag pairs (primary and secondary) for this chunk.
                    Existing tags: {existing_tags}
                    Consider using existing tags if they fit well. If necessary, create new tags.

                    Identify all technical details, including:
                    - Definitions
                    - Theorems
                    - Lemmas
                    - Proofs
                    - Equations
                    - Algorithms
                    - Pseudocode
                    - Data structures
                    - Experimental protocols

                    For each technical detail, provide:
                    - Full content
                    - Explanations of all variables, terms, and symbols used
                    - Context and significance within the paper
                    - Dependencies on other concepts or prior knowledge

                    Your response must be a valid JSON object with no additional text before or after.
                    Each key in the JSON object should be a primary tag, containing a nested object of secondary tags.
                    Each secondary tag should have the following fields:
                    - "analysis": Detailed analysis covering all points mentioned above
                    - "technical_details": Array of objects, each with "type", "content", "explanation", and "dependencies" fields
                    - "key_findings": Array of strings, each representing a crucial finding or insight
                    - "methodologies": Array of strings describing methods used
                    - "novel_contributions": Array of strings highlighting innovative aspects
                    - "future_directions": Array of strings suggesting potential future research

                    Example format:
                    {{
                        "Primary Tag 1": {{
                            "Secondary Tag 1": {{
                                "analysis": "Extremely detailed analysis covering all requested points...",
                                "technical_details": [
                                    {{
                                        "type": "theorem",
                                        "content": "Full theorem statement...",
                                        "explanation": "Detailed explanation of all terms and symbols...",
                                        "dependencies": "Prior knowledge required, related concepts..."
                                    }},
                                    {{
                                        "type": "algorithm",
                                        "content": "Full algorithm description or pseudocode...",
                                        "explanation": "Explanation of each step, variables used...",
                                        "dependencies": "Data structures or concepts the algorithm relies on..."
                                    }}
                                ],
                                "key_findings": [
                                    "Detailed key finding 1...",
                                    "Detailed key finding 2..."
                                ],
                                "methodologies": [
                                    "Detailed description of methodology 1...",
                                    "Detailed description of methodology 2..."
                                ],
                                "novel_contributions": [
                                    "Detailed description of novel contribution 1...",
                                    "Detailed description of novel contribution 2..."
                                ],
                                "future_directions": [
                                    "Detailed suggestion for future research 1...",
                                    "Detailed suggestion for future research 2..."
                                ]
                            }}
                        }}
                    }}

                    Respond only with the JSON object:
                    """
                    response = await self.client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=7500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    try:
                        result = json.loads(response.content[0].text)
                        logger.info(f"Chunk {chunk_number}/{total_chunks} analyzed successfully, {len(result)} primary tags generated")
                        self.chunk_cache[cache_key] = result
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response for chunk {chunk_number}/{total_chunks}: {str(e)}")
                        logger.error(f"Response content: {response.content[0].text}")
                        return None
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"API call failed for chunk {chunk_number}/{total_chunks}, retrying in {RETRY_DELAY} seconds. Error: {str(e)}")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"API call failed after {MAX_RETRIES} attempts for chunk {chunk_number}/{total_chunks}. Error: {str(e)}")
                        return None

    async def process_local_pdf(self, file_path: str):
        logger.info(f"Processing local PDF: {file_path}")
        try:
            # Read the PDF file
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
            
            # Extract metadata
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown Title')
            abstract = self.extract_abstract(doc)
            
            # Generate a unique ID for the paper
            paper_id = f"local_{hashlib.md5(file_path.encode()).hexdigest()}"
            
            # Create a paper object
            paper = {
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'pdf_url': file_path  # Use local file path as URL
            }
            
            # Process the paper
            await self.process_paper(paper)
            
            logger.info(f"Local PDF processed successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error processing local PDF {file_path}: {str(e)}")

    def extract_abstract(self, doc):
        # Attempt to extract abstract from the first page
        first_page = doc[0]
        text = first_page.get_text()
        
        # Look for common abstract indicators
        abstract_start = text.lower().find('abstract')
        if abstract_start != -1:
            abstract_end = text.lower().find('introduction', abstract_start)
            if abstract_end == -1:
                abstract_end = len(text)
            return text[abstract_start:abstract_end].strip()
        
    def _get_cache_key(self, paper_id: str, chunk: str) -> str:
        """Generate a unique cache key for a paper chunk."""
        return hashlib.md5(f"{paper_id}:{chunk[:100]}".encode()).hexdigest()


class GracefulExit(SystemExit):
    pass


def signal_handler(signum, frame):
    raise GracefulExit()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
    
class CustodianSystem:
    def __init__(self, api_key: str, db_path: str, num_haiku_agents: int = 3, api_timeout: int = 300):
        self.api_key = api_key
        self.db_path = db_path
        self.num_haiku_agents = num_haiku_agents
        self.clients = [AsyncAnthropic(api_key=api_key) for _ in range(num_haiku_agents)]
        self.db = duckdb.connect(db_path)
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour
        self.api_semaphore = asyncio.Semaphore(1)  # Allow only 1 API call at a time
        self.exit_event = asyncio.Event()
        self.encoder = tiktoken.get_encoding("cl100k_base")  # For token counting
        self.max_tokens = 8000  # Maximum allowed tokens for Claude 3.5 Sonnet
        self.max_batch_tokens = 4000  # Target token count for each batch
        self.api_timeout = api_timeout  # Timeout for API calls in seconds
        logger.info(f"CustodianSystem initialized with {num_haiku_agents} Haiku agents and database at {db_path}")

        # Set up signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP):
            signal.signal(sig, self.signal_handler)

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
        self.exit_event.set()
        raise GracefulExit()

    async def consolidate_database(self):
        logger.info("Starting database consolidation process")
        tables = ["tag_summaries", "technical_details", "key_findings", "methodologies", "novel_contributions", "future_directions"]
        
        total_consolidated = 0
        start_time = time.time()
        try:
            for table_index, table in enumerate(tables):
                if self.exit_event.is_set():
                    logger.info("Exit event detected. Stopping consolidation process.")
                    break

                logger.info(f"Starting consolidation of table: {table} ({table_index + 1}/{len(tables)})")
                rows = self.fetch_rows(table)
                initial_count = len(rows)
                logger.info(f"Fetched {initial_count} rows from table {table}")
                
                consolidated_rows = await self.process_rows(table, rows)
                final_count = len(consolidated_rows)
                consolidated_count = initial_count - final_count
                total_consolidated += consolidated_count
                
                logger.info(f"Table {table} consolidation complete: {consolidated_count} rows consolidated. Initial: {initial_count}, Final: {final_count}")
                
                self.update_database(table, consolidated_rows)
                
                elapsed_time = time.time() - start_time
                avg_time_per_table = elapsed_time / (table_index + 1)
                estimated_time_remaining = avg_time_per_table * (len(tables) - table_index - 1)
                logger.info(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
            
            logger.info(f"Database consolidation completed. Total rows consolidated across all tables: {total_consolidated}")
        except GracefulExit:
            logger.info("Graceful exit initiated during consolidation process.")
        except Exception as e:
            logger.error(f"An error occurred during consolidation: {str(e)}")
        finally:
            self.cleanup()

    def fetch_rows(self, table: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching rows from table: {table}")
        query = f"SELECT * FROM {table}"
        result = self.db.execute(query).fetchall()
        columns = [desc[0] for desc in self.db.description]
        rows = [dict(zip(columns, row)) for row in result]
        logger.info(f"Fetched {len(rows)} rows from table {table}")
        return rows

    async def process_rows(self, table: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Processing {len(rows)} rows from table {table}")
        batches = self.create_batches(rows)
        logger.info(f"Created {len(batches)} batches for processing")

        tasks = []
        for i, batch in enumerate(batches):
            tasks.append(self.process_batch(table, batch, self.clients[i % len(self.clients)], i))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        consolidated = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {str(result)}")
            else:
                consolidated.extend(result)

        logger.info(f"Table {table}: Processed {len(rows)} rows, resulting in {len(consolidated)} rows")
        return consolidated

    def create_batches(self, rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for row in rows:
            row_tokens = len(self.encoder.encode(json.dumps(row, cls=DateTimeEncoder)))
            if current_tokens + row_tokens > self.max_batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(row)
            current_tokens += row_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def process_batch(self, table: str, batch: List[Dict[str, Any]], client: AsyncAnthropic, batch_num: int) -> List[Dict[str, Any]]:
        if self.exit_event.is_set():
            logger.info(f"Exit event detected. Skipping batch {batch_num} of table {table}")
            return batch

        logger.info(f"Processing batch {batch_num} for table {table} with {len(batch)} rows")
        prompt = self.create_prompt(table, batch)
        cache_key = self.get_cache_key(prompt)
        
        if cache_key in self.cache:
            logger.info(f"Cache hit for batch {batch_num} of table {table}")
            return self.cache[cache_key]
        
        logger.info(f"Cache miss for batch {batch_num} of table {table}. Sending request to API.")
        async with self.api_semaphore:  # Use semaphore for rate limiting
            try:
                logger.debug(f"Acquired API semaphore for batch {batch_num} of table {table}")
                start_time = time.time()
                response = await asyncio.wait_for(
                    client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=self.api_timeout
                )
                elapsed_time = time.time() - start_time
                logger.debug(f"Received API response for batch {batch_num} of table {table} in {elapsed_time:.2f} seconds")
                consolidated = json.loads(response.content[0].text)
                
                # Handle datetime objects
                for item in consolidated:
                    for key, value in item.items():
                        if isinstance(value, str) and value.endswith(('Z', '+00:00')) and 'T' in value:
                            try:
                                item[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except ValueError:
                                logger.warning(f"Could not parse datetime string: {value}")
                
                # Verify consolidation
                if len(consolidated) > len(batch):
                    logger.warning(f"Consolidation resulted in more rows ({len(consolidated)}) than original ({len(batch)}). Using original batch.")
                    return batch
                
                consolidation_ratio = len(consolidated) / len(batch)
                if consolidation_ratio < 0.5:  # If more than 50% of rows were consolidated
                    logger.warning(f"High consolidation ratio detected: {consolidation_ratio:.2f}. Using original batch.")
                    return batch
                
                self.cache[cache_key] = consolidated
                logger.info(f"Successfully processed and cached batch {batch_num} for table {table}. Consolidated from {len(batch)} to {len(consolidated)} rows.")
                return consolidated
            except asyncio.TimeoutError:
                logger.error(f"API request timed out after {self.api_timeout} seconds for batch {batch_num} of table {table}")
                return batch
            except Exception as e:
                logger.error(f"Error processing batch {batch_num} for table {table}: {str(e)}")
                return batch
            finally:
                logger.debug(f"Released API semaphore for batch {batch_num} of table {table}")
                await asyncio.sleep(1)  # Add a 1-second delay between API calls

    def create_prompt(self, table: str, batch: List[Dict[str, Any]]) -> str:
        logger.debug(f"Creating prompt for table {table} with {len(batch)} rows")
        prompt = f"""
        Analyze the following rows from the {table} table:

        {json.dumps(batch, indent=2, cls=DateTimeEncoder)}

        Your task is to consolidate these rows ONLY if there is clear redundancy or duplication. Follow these strict guidelines:

        1. Preserve unique information: Each row likely contains some unique insights or details. Do not lose this information.
        2. Identify truly redundant data: Only combine information if it's essentially identical across multiple rows.
        3. Maintain granularity: It's better to keep rows separate if there's any doubt about their similarity.
        4. Preserve context: Ensure that any consolidation doesn't lose the context in which information was presented.
        5. Respect different sources or perspectives: If rows represent different viewpoints or sources, keep them separate.

        Consolidation rules:
        - Do NOT combine rows unless they are nearly identical in content and meaning.
        - Preserve all unique identifiers (e.g., summary_id, detail_id).
        - For text fields, only combine if the content is substantially the same. If in doubt, keep separate.
        - For list fields (e.g., key_findings), only remove exact duplicates.
        - If rows have different timestamps or metadata, consider keeping them separate.

        Your response should be a list of dictionaries. Each dictionary should represent either:
        a) An original, unconsolidated row if it contains unique information
        b) A consolidated row that combines only truly redundant information

        Aim to consolidate as little as possible. It's perfectly acceptable to return all original rows if they each contain unique information.

        Return your response as a valid JSON array of objects, with no additional text before or after.
        """
        logger.debug(f"Created conservative consolidation prompt for table {table}")
        return prompt

    def update_database(self, table: str, consolidated_rows: List[Dict[str, Any]]):
        logger.info(f"Updating database for table {table} with {len(consolidated_rows)} consolidated rows")
        try:
            self.db.begin()  # Start a transaction
            for i, row in enumerate(consolidated_rows):
                if self.exit_event.is_set():
                    logger.info("Exit event detected. Stopping database update.")
                    break

                update_query = f"UPDATE {table} SET "
                update_columns = []
                values = []
                for key, value in row.items():
                    if key.lower() != 'id':  # Ignore the 'id' column
                        if isinstance(value, datetime):
                            update_columns.append(f"{key} = ?")
                            values.append(value.isoformat())
                        else:
                            update_columns.append(f"{key} = ?")
                            values.append(value)
                
                if not update_columns:
                    logger.warning(f"No updateable columns found for row {i+1} in table {table}")
                    continue
                
                update_query += ", ".join(update_columns)
                update_query += f" WHERE id = ?"
                values.append(row['id'])  # Add the id as the last parameter
                
                self.db.execute(update_query, values)
                logger.debug(f"Updated row {i+1}/{len(consolidated_rows)} in table {table}")
            
            if not self.exit_event.is_set():
                self.db.commit()
                logger.info(f"Successfully committed updates for {len(consolidated_rows)} rows in table {table}")
            else:
                self.db.rollback()
                logger.info(f"Rolled back changes for table {table} due to exit event")
        except Exception as e:
            logger.error(f"Error updating database for table {table}: {str(e)}")
            self.db.rollback()
            logger.info(f"Rolled back changes for table {table} due to error")

    def get_cache_key(self, prompt: str) -> str:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    def cleanup(self):
        logger.info("Starting cleanup process")
        try:
            self.db.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        logger.info("Cleanup process completed")

    def close(self):
        self.cleanup()

async def main():
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    orchestrator = Orchestrator(api_key)
    
    def cleanup():
        orchestrator.close_db()
        logger.info("Cleanup completed")

    try:
        # await orchestrator.execute_system("Stochastic Processes")
        # Process local PDFs
        local_pdfs = [
            "real_analysis.pdf"
            # Add more local PDF paths as needed
        ]
        for pdf_path in local_pdfs:
            if os.path.exists(pdf_path):
                await orchestrator.process_local_pdf(pdf_path)
            else:
                logger.warning(f"Local PDF not found: {pdf_path}")
    #     num_agents = 5
    #     custodian = CustodianSystem(api_key, "research_papers_v15.db", num_agents)
    #     try:
    #         await custodian.consolidate_database()
    #     except GracefulExit:
    #         logger.info("Graceful exit completed.")
    #     finally:
    #         custodian.close()
    # except GracefulExit:
        logger.info("Received termination signal. Cleaning up...")
        cleanup()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        cleanup()
    finally:
        cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Script started")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main())
    except GracefulExit:
        pass  # The cleanup has already been done in the main function
    except Exception as e:
        logger.error(f"An unhandled error occurred: {str(e)}")
    
    logger.info("Script completed")