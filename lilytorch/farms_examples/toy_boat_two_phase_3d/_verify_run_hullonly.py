import os
from _verify_run_small import VerifySmall


class VerifyHullOnly(VerifySmall):
    def __init__(self):
        super().__init__()
        boat_sdf = os.path.join(self.data_folder, 'toy_boat_hullonly.sdf')
        self.animats_pars[0]["sdf_file"]   = boat_sdf
        self.animats_pars[0]["model_name"] = "toy_boat_hullonly"


if __name__ == "__main__":
    print("=== VERIFY HULL-ONLY (single body, S=0.1) ===", flush=True)
    VerifyHullOnly().run()
