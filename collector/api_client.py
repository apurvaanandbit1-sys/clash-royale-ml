import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CR_API_KEY")

if API_KEY is None:
    raise Exception("CR_API_KEY not found inside .env")

BASE_URL = "https://api.clashroyale.com/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}


try:
    from config.settings import MAX_RETRIES, RETRY_DELAY
except ImportError:
    MAX_RETRIES = 5
    RETRY_DELAY = 3


class ClashRoyaleAPI:
    def __init__(self, delay=0.25):
        self.delay = delay

    def _request(self, endpoint):
        """
        Makes a safe request to the Clash Royale API.

        Automatically retries with exponential backoff if:
            - Rate limited (429)
            - Temporary server errors
            - Network Exceptions
        Raises an exception if retries are exhausted.
        """

        url = BASE_URL + endpoint
        retries = 0
        current_delay = RETRY_DELAY

        while retries < MAX_RETRIES:
            try:
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=15
                )

                if response.status_code == 200:
                    time.sleep(self.delay)
                    return response.json()

                elif response.status_code == 429:
                    print(f"Rate limited [429]. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    retries += 1
                    current_delay *= 2  # Exponential backoff

                elif response.status_code >= 500:
                    print(f"Server Error {response.status_code}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    retries += 1
                    current_delay *= 2

                else:
                    print(f"Request Failed [{response.status_code}] : {endpoint}")
                    return None

            except requests.exceptions.RequestException as e:
                print(f"Network Error: {e}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                retries += 1
                current_delay *= 2

        raise Exception(f"API request failed after {MAX_RETRIES} retries for endpoint: {endpoint}")

    # --------------------------------------------------------

    def get_player(self, player_tag):

        tag = player_tag.replace("#", "%23")

        return self._request(
            f"/players/{tag}"
        )

    # --------------------------------------------------------

    def get_battle_log(self, player_tag):

        tag = player_tag.replace("#", "%23")

        return self._request(
            f"/players/{tag}/battlelog"
        )

    # --------------------------------------------------------

    def search_clans(self, name, limit=10):

        return self._request(
            f"/clans?name={name}&limit={limit}"
        )

    # --------------------------------------------------------

    def get_clan_members(self, clan_tag):

        tag = clan_tag.replace("#", "%23")

        return self._request(
            f"/clans/{tag}/members"
        )
    
        # --------------------------------------------------------

    def get_cards(self):

        data = self._request("/cards")

        if data is None:
            return []

        return data.get("items", [])