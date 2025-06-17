import asyncio
import logging
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetBotCallbackAnswerRequest
from telethon.tl.functions.channels import JoinChannelRequest, LeaveChannelRequest, GetParticipantRequest
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.errors import (
    FloodWaitError, UserAlreadyParticipantError, UserNotParticipantError,
    InviteRequestSentError, ChatAdminRequiredError, ChannelPrivateError
)
from datetime import datetime
import json
import os
import traceback
from urllib.parse import urlparse

class TelegramConfig:
    # API Configuration
    API_ID = <Enter your API ID here>
    API_HASH = "Enter your API Hash here"
    BOT_USERNAME = '@en_SearchBot'
    SESSION_NAME = 'search_bot_session'
    
    # Operation Limits
    REQUEST_DELAY = 3
    MAX_CLICKS = 10
    MAX_GROUPS_TO_JOIN = 15
    MAX_KEYWORD_MATCHES = 50
    MAX_SCAN_MESSAGES = 300
    USE_FAST_FORWARD = True
    
    # File paths
    LINKS_OUTPUT_FILE = 'scraped_links.txt'
    OUTPUT_FILE = 'channel_messages.json'
    
    # Search parameters
    KEYWORDS = ['global warming', 'climate change']

class LoggerSetup:
    def __init__(self, name='TelegramScraper'):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Set the logging level
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create and set up file handler
        file_handler = logging.FileHandler('scraper.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Create and set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger

class TelegramScraper:
    def __init__(self):
        self.config = TelegramConfig()
        self.logger = LoggerSetup().get_logger()
        self.seen_keywords = {}
        self.scraped_data = self._load_scraped_data()
        self.search_bot_entity = None
        self.next_button_clicks = 0
        self.client = None
    
    def _load_scraped_data(self):
        if os.path.exists(self.config.OUTPUT_FILE):
            try:
                with open(self.config.OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Error loading {self.config.OUTPUT_FILE}. Starting with empty data.")
                return {}
        return {}
    
    def _save_scraped_data(self):
        try:
            with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Successfully saved data to {self.config.OUTPUT_FILE}")
        except Exception as e:
            self.logger.error(f"Error saving scraped data: {str(e)}")
    
    @staticmethod
    def extract_username_from_link(link):
        path = urlparse(link).path.strip('/')
        return path.split('/')[0] if '/' in path else path
    
    async def is_participant(self, entity):
        try:
            result = await self.client(GetParticipantRequest(channel=entity, participant='me'))
            self.logger.debug(f"Successfully checked participation status for entity {entity}")
            return result is not None
        except UserNotParticipantError:
            self.logger.debug(f"Not a participant in entity {entity}")
            return False
        except Exception as e:
            self.logger.error(f"Error checking participant status: {str(e)}")
            return False
    
    async def safe_leave(self, entity, channel_id):
        if await self.is_participant(entity):
            try:
                await self.client(LeaveChannelRequest(entity))
                self.logger.info(f"[LEFT] Successfully left channel {channel_id}")
            except Exception as e:
                self.logger.error(f"[LEAVE ERROR] {channel_id}: {str(e)}")
        else:
            self.logger.debug(f"[SKIP LEAVE] {channel_id}: Not a participant")
    
    async def extract_links_from_message(self, message):
        if message.entities:
            try:
                links = [e.url for e in message.entities if hasattr(e, 'url') and e.url]
                with open(self.config.LINKS_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    for link in links:
                        f.write(link + '\n')
                self.logger.debug(f"Extracted {len(links)} links from message {message.id}")
            except Exception as e:
                self.logger.error(f"Error extracting links from message: {str(e)}")
    
    async def find_and_click_pagination_button(self, message):
        if self.next_button_clicks >= self.config.MAX_CLICKS:
            self.logger.info("Maximum pagination clicks reached")
            return False
        
        if not message.reply_markup:
            self.logger.debug("No reply markup found in message")
            return False
        
        self.logger.debug(f"Checking pagination buttons in message {message.id}")
        for row in message.buttons:
            for button in row:
                if button.text in ['➡️', 'Next', '⏩']:
                    try:
                        await self.client(GetBotCallbackAnswerRequest(
                            peer=message.peer_id,
                            msg_id=message.id,
                            data=button.data
                        ))
                        self.next_button_clicks += 1
                        self.logger.info(f"Clicked pagination button: {button.text} (Click count: {self.next_button_clicks})")
                        await asyncio.sleep(self.config.REQUEST_DELAY)
                        return True
                    except Exception as e:
                        self.logger.error(f"Error clicking pagination button: {str(e)}")
                        return False
        return False
    
    async def get_or_reuse_last_bot_message(self, entity, keyword):
        try:
            async for msg in self.client.iter_messages(entity, from_user=entity, limit=20):
                if keyword.lower() in msg.message.lower():
                    self.logger.debug(f"Reusing existing message for keyword '{keyword}'")
                    return msg
            
            self.logger.info(f"Sending new message with keyword '{keyword}'")
            await self.client.send_message(entity, keyword)
            await asyncio.sleep(2)
            return (await self.client.get_messages(entity, limit=1))[0]
        except Exception as e:
            self.logger.error(f"Error getting bot message: {str(e)}")
            raise
    
    async def process_keyword(self, keyword):
        try:
            if keyword in self.seen_keywords:
                message = self.seen_keywords[keyword]
                self.logger.debug(f"Using cached message for keyword '{keyword}'")
            else:
                message = await self.get_or_reuse_last_bot_message(self.search_bot_entity, keyword)
                self.seen_keywords[keyword] = message
                self.logger.info(f"Processing new keyword '{keyword}'")
            
            await self.extract_links_from_message(message)
            while await self.find_and_click_pagination_button(message):
                message = await self.client.get_messages(self.search_bot_entity, ids=message.id)
                await self.extract_links_from_message(message)
        
        except Exception as e:
            self.logger.error(f"Error processing keyword '{keyword}': {str(e)}")
    
    async def scrape_messages_from_channel(self, link):
        username = self.extract_username_from_link(link)
        if username in self.scraped_data:
            self.logger.info(f"Skipping already scraped channel: {username}")
            return True
        
        entity = None
        try:
            try:
                entity = await self.client.get_entity(username)
            except ValueError as e:
                self.logger.error(f"[SCRAPE ERROR] {username}: {str(e)}")
                return False
            
            try:
                await self.client(JoinChannelRequest(entity))
                self.logger.info(f"[JOINED] Successfully joined {username}")
            except UserAlreadyParticipantError:
                self.logger.debug(f"[EXISTING] Already a member of {username}")
            except FloodWaitError as e:
                self.logger.warning(f"FloodWaitError for {username}, waiting {e.seconds} seconds")
                await asyncio.sleep(e.seconds)
                return False
            except (InviteRequestSentError, ChatAdminRequiredError, ChannelPrivateError) as e:
                self.logger.info(f"[SKIP] {username}: {str(e)} - Channel requires approval or is private")
                return False
            except Exception as e:
                self.logger.error(f"[JOIN ERROR] {username}: {str(e)}")
                return False
            
            await asyncio.sleep(self.config.REQUEST_DELAY)
            
            try:
                messages = await self.client(GetHistoryRequest(
                    peer=entity,
                    limit=self.config.MAX_SCAN_MESSAGES,
                    offset_date=None,
                    offset_id=0,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))
                
                group_data = {"link": link, "messages": []}
                count = 0
                for msg in messages.messages:
                    if msg.sender_id and hasattr(msg, 'message') and msg.message:
                        if any(k.lower() in msg.message.lower() for k in self.config.KEYWORDS):
                            group_data["messages"].append({
                                "user_id": str(msg.sender_id),
                                "selftext": msg.message
                            })
                            count += 1
                            if count >= self.config.MAX_KEYWORD_MATCHES:
                                break
                
                if count:
                    self.scraped_data[username] = group_data
                    self.logger.info(f"[SAVED] {count} messages from {username}")
                else:
                    self.logger.info(f"[EMPTY] No matching messages in {username}")
                
                return True
            
            except ChatAdminRequiredError:
                self.logger.info(f"[SKIP] {username}: Admin privileges required")
                return False
            except Exception as e:
                self.logger.error(f"[SCRAPE ERROR] {username}: {str(e)}\n{traceback.format_exc()}")
                return False
        
        except Exception as e:
            self.logger.error(f"[SCRAPE ERROR] {username}: {str(e)}\n{traceback.format_exc()}")
            return False
        
        finally:
            if entity is not None:
                await self.safe_leave(entity, username)
                self._save_scraped_data()
    
    async def run(self):
        self.logger.info("Starting Telegram scraping process")
        try:
            async with TelegramClient(
                self.config.SESSION_NAME,
                self.config.API_ID,
                self.config.API_HASH
            ) as self.client:
                await self.client.start()
                self.logger.info("Successfully connected to Telegram")
                
                self.search_bot_entity = await self.client.get_entity(self.config.BOT_USERNAME)
                self.logger.info(f"Connected to search bot: {self.config.BOT_USERNAME}")
                
                for keyword in self.config.KEYWORDS[:2]:
                    self.logger.info(f"Processing keyword: {keyword}")
                    await self.process_keyword(keyword)
                
                if os.path.exists(self.config.LINKS_OUTPUT_FILE):
                    with open(self.config.LINKS_OUTPUT_FILE, 'r') as f:
                        links = list(set(l.strip() for l in f if l.strip()))
                    
                    self.logger.info(f"Found {len(links)} unique links to process")
                    count = 0
                    for link in links:
                        if count >= self.config.MAX_GROUPS_TO_JOIN:
                            self.logger.info("Reached maximum number of groups to join")
                            break
                        
                        success = await self.scrape_messages_from_channel(link)
                        if success:
                            count += 1
                            self.logger.info(f"Successfully processed {count}/{self.config.MAX_GROUPS_TO_JOIN} groups")
                
                self.logger.info("Scraping process completed successfully")
        
        except Exception as e:
            self.logger.error(f"Critical error in main execution: {str(e)}\n{traceback.format_exc()}")
            raise

if __name__ == '__main__':
    scraper = TelegramScraper()
    asyncio.run(scraper.run())
