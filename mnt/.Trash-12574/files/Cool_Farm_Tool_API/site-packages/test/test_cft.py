import pandas as pd
from pathlib import Path
import pytest
from tcc_cft_tool import CftTool, CftTemplate
from tcc_s3 import SSM


TEST_DIR_PATH = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def cft():
    ssm = SSM()
    app_key = ssm.get_parameter("ci.team-science.cft-api.app-key")
    input_df = CftTemplate().input_df_template
    cft_class = CftTool(input_df, app_key)
    cft_class.validate_request()
    cft_class.call_api()
    return cft_class


def test_api_transformation(cft):
   # check request and return
    assert all(cft.validation_result['validate_status'])


def test_return_summary(cft):
    pd.testing.assert_frame_equal(cft.parse_summary(), pd.read_csv(TEST_DIR_PATH / 'summary.csv'))


def test_return_detail(cft):
    pd.testing.assert_frame_equal(cft.parse_detail(), pd.read_csv(TEST_DIR_PATH / 'detail.csv'))
