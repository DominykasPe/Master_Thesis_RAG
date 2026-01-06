import pandas as pd
import time
from typing import Dict, List
import logging
import os
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime

class NASALessonsLearned:
    def __init__(self, max_workers=4, start_year=2000, end_year=None):
        """
        Initialize NASA Lessons Learned scraper.
        
        Args:
            max_workers: Number of parallel browser instances for scraping
            start_year: Start year for filtering lessons (default: 2000)
            end_year: End year for filtering lessons (default: current year)
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Number of parallel workers
        self.max_workers = max_workers
        self.start_year = start_year
        self.end_year = end_year if end_year else datetime.now().year
        
        # Setup Selenium with Firefox
        self.options = Options()
        self.options.add_argument('--headless')  # Run in headless mode
        
        # Base URL for NASA LLIS
        self.base_url = "https://llis.nasa.gov"
        
        # Build date ranges for scraping
        # NASA LLIS uses specific date range formats
        self.date_ranges = self._build_date_ranges()
        
        self.logger.info(f"Will scrape lessons from {self.start_year} to {self.end_year}")
        self.logger.info(f"Date ranges to process: {len(self.date_ranges)}")
        
        # CSV filename based on date range
        self.csv_filename = f'nasa_lessons_learned_{self.start_year}_{self.end_year}.csv'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(script_dir, self.csv_filename)
        
        # Check if CSV exists, create it with headers if it doesn't
        if not os.path.exists(self.csv_path):
            self.logger.info(f"Creating new CSV file: {self.csv_filename}")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = pd.DataFrame(columns=[
                    'url', 'subject', 'abstract', 'driving_event', 
                    'lessons_learned', 'recommendations', 'evidence',
                    'program_relation', 'program_phase', 
                    'mission_directorate', 'topics', 'date_range'
                ]).to_csv(f, index=False)
        else:
            self.logger.info(f"Appending to existing CSV file: {self.csv_filename}")
        
        # Create a driver for URL collection
        self.driver = webdriver.Firefox(options=self.options)
    
    def _build_date_ranges(self) -> List[str]:
        """
        Build date range parameters for NASA LLIS URLs.
        
        NASA LLIS uses formats like:
        - '2000-2003' for early years grouped together
        - '2004', '2005', etc. for individual years
        """
        date_ranges = []
        current_year = self.start_year
        
        # Handle early years grouped (2000-2003 is often grouped on LLIS)
        if current_year <= 2003 and self.end_year >= 2000:
            if self.start_year <= 2003:
                date_ranges.append('2000-2003')
                current_year = 2004
        
        # Add individual years from 2004 onwards
        while current_year <= self.end_year:
            date_ranges.append(str(current_year))
            current_year += 1
        
        return date_ranges

    def get_lessons_urls(self, max_pages_per_range: int = 50) -> List[tuple]:
        """
        Collect lesson URLs for all date ranges.
        
        Returns:
            List of tuples: (url, date_range)
        """
        all_lesson_urls = []
        
        self.logger.info(f"Starting to collect URLs across {len(self.date_ranges)} date ranges...")
        
        for date_range in self.date_ranges:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing date range: {date_range}")
            self.logger.info(f"{'='*50}")
            
            range_urls = []
            page = 1
            
            while page <= max_pages_per_range:
                try:
                    # Build URL with date filter
                    search_url = f"{self.base_url}/search?lesson_date={date_range}&page={page}"
                    self.logger.info(f"Collecting URLs from: {search_url}")
                    
                    self.driver.get(search_url)
                    
                    # Wait for the page to load
                    time.sleep(3)
                    
                    # Check if there are any lessons on this page
                    lesson_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/lesson/']")
                    
                    # If no lessons found, we've reached the end for this date range
                    if not lesson_elements:
                        self.logger.info(f"No more lessons found for {date_range} on page {page}. Moving to next date range.")
                        break
                    
                    # Get all lesson links
                    new_urls = [(elem.get_attribute('href'), date_range) for elem in lesson_elements]
                    range_urls.extend(new_urls)
                    
                    self.logger.info(f"Found {len(new_urls)} lessons on page {page} for {date_range}")
                    
                    # Check if we're on the last page by looking at the pagination controls
                    try:
                        pagination = self.driver.find_element(By.CSS_SELECTOR, ".pagination")
                        active_page = pagination.find_element(By.CSS_SELECTOR, ".active").text
                        page_numbers = [el.text for el in pagination.find_elements(By.CSS_SELECTOR, "li:not(.prev):not(.next) a")]
                        
                        if page_numbers and active_page == page_numbers[-1]:
                            self.logger.info(f"Reached the last page ({active_page}) for {date_range}.")
                            break
                    except Exception as pagination_error:
                        self.logger.debug(f"Could not determine pagination status: {pagination_error}")
                    
                    page += 1
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error on page {page} for {date_range}: {e}")
                    page += 1  # Try the next page
            
            self.logger.info(f"Collected {len(range_urls)} lessons for date range {date_range}")
            all_lesson_urls.extend(range_urls)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Finished collecting URLs. Total lessons found: {len(all_lesson_urls)}")
        self.logger.info(f"{'='*50}")
        
        return all_lesson_urls

    def _get_text(self, soup: BeautifulSoup, field_name: str) -> str:
        """Helper method to extract text from a field"""
        try:
            # Find the div with ember-view class that contains the field
            for div in soup.find_all('div', class_='ember-view'):
                # Look for h3 with the field name
                h3 = div.find('h3', string=lambda x: x and field_name in x)
                if h3:
                    # Get all text content after the h3
                    content = []
                    for sibling in h3.next_siblings:
                        if sibling.name == 'h3':  # Stop if we hit another h3
                            break
                        if hasattr(sibling, 'stripped_strings'):
                            content.extend(sibling.stripped_strings)
                        elif hasattr(sibling, 'string') and sibling.string:
                            content.append(sibling.string.strip())
                    return ' '.join(filter(None, content))
            
            return "None"
        except Exception as e:
            self.logger.error(f"Error extracting {field_name}: {e}")
            return "None"

    def _get_subject(self, soup: BeautifulSoup) -> str:
        """Special method to extract subject"""
        try:
            # Find the div with ember-view class that contains 'Subject'
            for div in soup.find_all('div', class_='ember-view'):
                h3 = div.find('h3', string='Subject')
                if h3:
                    # Look for the strong tag within em
                    em = div.find('em')
                    if em:
                        strong = em.find('strong')
                        if strong:
                            return strong.get_text(strip=True)
                    # Backup: get any text content after the h3
                    content = []
                    for sibling in h3.next_siblings:
                        if hasattr(sibling, 'stripped_strings'):
                            content.extend(sibling.stripped_strings)
                    if content:
                        return ' '.join(content)
            return "None"
        except Exception as e:
            self.logger.error(f"Error extracting subject: {e}")
            return "None"

    def extract_lesson_data(self, url_tuple: tuple) -> Dict:
        """
        Extract data from a single lesson URL.
        
        Args:
            url_tuple: Tuple of (url, date_range)
        """
        url, date_range = url_tuple
        driver = None
        try:
            # Create a new driver for this thread
            driver = webdriver.Firefox(options=self.options)
            self.logger.debug(f"Extracting data from {url}")
            driver.get(url)
            
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ember-view"))
                )
            except Exception as wait_error:
                self.logger.warning(f"Timeout waiting for page load on {url}, attempting to extract anyway")
            
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Get subject using the special method
            subject_text = self._get_subject(soup)
            
            data = {
                'url': url,
                'subject': subject_text,
                'abstract': self._get_text(soup, 'Abstract'),
                'driving_event': self._get_text(soup, 'Driving Event'),
                'lessons_learned': self._get_text(soup, 'Lesson(s) Learned'),
                'recommendations': self._get_text(soup, 'Recommendation(s)'),
                'evidence': self._get_text(soup, 'Evidence of Recurrence Control Effectiveness'),
                'program_relation': self._get_text(soup, 'Program Relation'),
                'program_phase': self._get_text(soup, 'Program/Project Phase'),
                'mission_directorate': self._get_text(soup, 'Mission Directorate(s)'),
                'topics': self._get_text(soup, 'Topic(s)'),
                'date_range': date_range
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return {
                'url': url,
                'subject': "Error",
                'abstract': "Error",
                'driving_event': "Error",
                'lessons_learned': "Error",
                'recommendations': "Error",
                'evidence': "Error",
                'program_relation': "Error",
                'program_phase': "Error",
                'mission_directorate': "Error",
                'topics': "Error",
                'date_range': date_range
            }
        finally:
            if driver:
                driver.quit()
    
    def save_to_csv(self, data: Dict):
        """Save a single lesson to CSV"""
        try:
            pd.DataFrame([data]).to_csv(
                self.csv_path, 
                mode='a', 
                header=False, 
                index=False,
                encoding='utf-8'
            )
            self.logger.debug(f"Saved data for {data['url']} to CSV")
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}")
    
    def collect_all_lessons(self) -> pd.DataFrame:
        self.logger.info("Starting collection of all lessons...")
        self.logger.info(f"Date range: {self.start_year} to {self.end_year}")
        
        lesson_url_tuples = self.get_lessons_urls()
        total_urls = len(lesson_url_tuples)
        self.logger.info(f"Found {total_urls} lessons to process")
        
        # Check if we already have some of these URLs in the CSV
        try:
            existing_df = pd.read_csv(self.csv_path)
            existing_urls = set(existing_df['url'].tolist())
            lesson_url_tuples = [(url, date_range) for url, date_range in lesson_url_tuples if url not in existing_urls]
            self.logger.info(f"After filtering already processed URLs, {len(lesson_url_tuples)} lessons remain to be processed")
        except Exception as e:
            self.logger.warning(f"Could not check for existing URLs: {e}")
        
        if not lesson_url_tuples:
            self.logger.info("No new lessons to process!")
            return pd.read_csv(self.csv_path)
        
        # Process lessons in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(self.extract_lesson_data, url_tuple): url_tuple for url_tuple in lesson_url_tuples}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url), 1):
                url_tuple = future_to_url[future]
                url, date_range = url_tuple
                try:
                    data = future.result()
                    self.save_to_csv(data)
                    self.logger.info(f"Processed {i}/{len(lesson_url_tuples)}: {url} [{date_range}]")
                    if data['subject'] and data['subject'] != "Error":
                        self.logger.info(f"Subject: {data['subject'][:100]}...")
                except Exception as exc:
                    self.logger.error(f"Error processing {url}: {exc}")
        
        self.logger.info(f"Finished collecting lessons")
        return pd.read_csv(self.csv_path)
    
    def __del__(self):
        """Clean up the Selenium driver"""
        if hasattr(self, 'driver'):
            self.driver.quit()


if __name__ == "__main__":
    # Download all lessons from 2000 to current year
    # You can adjust the date range and number of parallel workers
    scraper = NASALessonsLearned(
        max_workers=4,
        start_year=2000,
        end_year=None  # None = current year
    )
    df = scraper.collect_all_lessons()
    print(f"\nTotal lessons collected: {len(df)}")