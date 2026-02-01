
import pandas as pd
import time
import random
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import argparse
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SpringerScraper:
    def __init__(self, input_csv: str, output_csv: str = None):
        self.input_path = Path(input_csv)
        if output_csv:
            self.output_path = Path(output_csv)
        else:
            self.output_path = self.input_path.parent / f"{self.input_path.stem}_with_abstracts.csv"
        
        self.driver = None

    def setup_driver(self):
        """Initializes the Chrome WebDriver."""
        logger.info("Initializing Selenium WebDriver...")
        options = Options()
        # options.add_argument("--headless") # Commented out to let user see progress as per possible preference, can be enabled.
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            logger.info("WebDriver initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def close_driver(self):
        """Closes the WebDriver."""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed.")

    def accept_cookies(self):
        """Attempts to accept cookies."""
        try:
            # Common cookie banner selectors for Springer/Nature
            cookie_selectors = [
                "button[data-cc-action='accept']",
                ".cc-banner .cc-btn.cc-allow",
                "button.osano-cm-accept-all",
                "button[data-test='cookie-banner-accept-btn']",
                "#onetrust-accept-btn-handler"
            ]
            
            for selector in cookie_selectors:
                try:
                    element = WebDriverWait(self.driver, 4).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    element.click()
                    logger.info(f"Clicked cookie banner: {selector}")
                    time.sleep(2) # Wait for banner to disappear
                    return
                except:
                    continue
        except Exception as e:
            logger.debug(f"No cookie banner found or clicked: {e}")

    def get_abstract(self, url: str) -> str:
        """Navigates to the URL and extracts the abstract."""
        if not url or pd.isna(url):
            return ""

        try:
            logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            
            # Random delay to mimic human behavior
            delay = random.uniform(0, 1)
            logger.info(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)

            self.accept_cookies()

            abstract_text = ""
            
            # Try different selectors common on Springer/Nature
            selectors = [
                "#Abs1-content",
                ".c-article-section__content",
                "section[data-title='Abstract'] .c-article-section__content",
                "div.c-article-section__content",
                "#Abs1",
                "section.Abstract",
                ".Abstract"
            ]
            
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element:
                        text = element.text.strip()
                        if text:
                            abstract_text = text
                            logger.info(f"Abstract found with selector: {selector}")
                            break
                except:
                    continue
            
            if abstract_text:
                logger.info("Abstract found.")
            else:
                logger.warning("Abstract NOT found on page.")
                # Save page source for debugging
                sanitized_name = re.sub(r'[^\w\-_\. ]', '_', url[-30:])
                debug_file = Path(f"debug_{sanitized_name}.html")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                logger.info(f"Saved page source to {debug_file}")
                
            return abstract_text

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""

    def process(self):
        """Main processing loop."""
        if not self.input_path.exists():
            logger.error(f"Input file not found: {self.input_path}")
            return

        # Load input or resume from output if exists
        if self.output_path.exists():
            logger.info(f"Found existing output file. Resuming from {self.output_path}")
            df = pd.read_csv(self.output_path)
        else:
            logger.info(f"Reading input file: {self.input_path}")
            df = pd.read_csv(self.input_path)
            # Ensure Abstract column exists
            if 'Abstract' not in df.columns:
                df['Abstract'] = ""

        try:
            self.setup_driver()
            
            # Identify rows to process (where Abstract is empty and URL exists)
            # handle 'Abstract' column being NaN or empty string
            mask = (df['Abstract'].isna() | (df['Abstract'] == "")) & (df['URL'].notna()) & (df['URL'] != "")
            total_to_process = mask.sum()
            
            logger.info(f"Found {total_to_process} entries to process.")
            
            count = 0
            for index, row in df[mask].iterrows():
                url = row['URL']
                
                abstract = self.get_abstract(url)
                
                # Update DataFrame using .at for safety
                df.at[index, 'Abstract'] = abstract
                
                count += 1
                
                # Save periodically
                if count % 5 == 0:
                    logger.info(f"Saving progress ({count} processed)...")
                    df.to_csv(self.output_path, index=False)
            
            # Final save
            df.to_csv(self.output_path, index=False)
            logger.info(f"Completed! processed {count} entries. Output saved to {self.output_path}")

        except KeyboardInterrupt:
            logger.info("Process interrupted by user. Saving progress...")
            df.to_csv(self.output_path, index=False)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            # Try to save whatever we have
            df.to_csv(self.output_path, index=False)
        finally:
            self.close_driver()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Springer abstracts from a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    
    scraper = SpringerScraper(args.input_csv)
    scraper.process()
