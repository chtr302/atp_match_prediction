import pandas as pd
import numpy as np

class TennisEloModel:
    def __init__(self, k_factor=32, surface_weight=0.5, start_elo=1500):
        """
        Args:
            k_factor (int): Hệ số K - độ nhạy của điểm Elo (K càng lớn, điểm càng biến động mạnh).
            surface_weight (float): Trọng số của Elo mặt sân (0.0 đến 1.0).
                                    Nếu 0.0: Chỉ dùng Elo tổng quát.
                                    Nếu 1.0: Chỉ dùng Elo mặt sân.
            start_elo (int): Điểm khởi đầu cho người mới.
        """
        self.k_factor = k_factor
        self.surface_weight = surface_weight
        self.start_elo = start_elo
        
        # Lưu trữ Elo tổng quát: {player_id: elo_score}
        self.overall_elo = {}
        
        # Lưu trữ Elo theo mặt sân: {player_id: {surface: elo_score}}
        self.surface_elo = {}

    def _get_elo(self, player_id, surface):
        """Lấy điểm Elo hiện tại của cầu thủ (kết hợp Overall và Surface)."""
        # 1. Lấy Overall Elo
        overall = self.overall_elo.get(player_id, self.start_elo)
        
        # 2. Lấy Surface Elo
        if player_id not in self.surface_elo:
            self.surface_elo[player_id] = {}
        s_elo = self.surface_elo[player_id].get(surface, self.start_elo)
        
        # 3. Tính điểm kết hợp (Weighted Average)
        # Công thức: Final = (1 - weight) * Overall + weight * Surface
        final_elo = (1 - self.surface_weight) * overall + self.surface_weight * s_elo
        return final_elo, overall, s_elo

    def _update_elo(self, w_id, l_id, surface):
        """Cập nhật điểm Elo sau mỗi trận đấu."""
        # Lấy điểm hiện tại
        w_final, w_over, w_surf = self._get_elo(w_id, surface)
        l_final, l_over, l_surf = self._get_elo(l_id, surface)
        
        # Tính xác suất thắng dự kiến (Expected Probability) theo công thức chuẩn Elo
        # E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
        prob_w = 1 / (1 + 10 ** ((l_final - w_final) / 400))
        prob_l = 1 - prob_w
        
        # Tính lượng điểm thay đổi (Delta)
        # Winner nhận được: K * (1 - Prob_Win)
        # Loser bị trừ: K * (0 - Prob_Lose) = - K * Prob_Lose
        delta = self.k_factor * (1 - prob_w)
        
        # Cập nhật Overall Elo
        self.overall_elo[w_id] = w_over + delta
        self.overall_elo[l_id] = l_over - delta
        
        # Cập nhật Surface Elo
        # Lưu ý: Surface Elo cũng dùng delta này (hoặc có thể dùng K riêng, nhưng để đơn giản ta dùng chung)
        if w_id not in self.surface_elo: self.surface_elo[w_id] = {}
        if l_id not in self.surface_elo: self.surface_elo[l_id] = {}
        
        self.surface_elo[w_id][surface] = w_surf + delta
        self.surface_elo[l_id][surface] = l_surf - delta

    def fit_transform(self, df):
        """
        Chạy mô hình trên toàn bộ dữ liệu lịch sử để sinh ra feature Elo.
        QUAN TRỌNG: Dữ liệu phải được sắp xếp theo thời gian trước!
        """
        print(f"--- ĐANG TÍNH TOÁN ELO (K={self.k_factor}, Surface Weight={self.surface_weight}) ---")
        
        # Đảm bảo dữ liệu đã sort
        if 'tourney_date' in df.columns:
            df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
            
        elo_diffs = []
        p1_elos = []
        p2_elos = []
        
        # Duyệt qua từng trận đấu
        for idx, row in df.iterrows():
            # Xác định ai là P1, ai là P2 (từ dữ liệu đã restructure)
            # Vì logic update cần biết ai thắng ai thua thực tế, ta cần dựa vào TARGET
            p1_id = row['p1_id']
            p2_id = row['p2_id']
            surface = row['surface']
            target = row['target'] # 1 nếu P1 thắng, 0 nếu P2 thắng
            
            # Lấy Elo TRƯỚC trận đấu (để làm feature dự đoán)
            p1_final, _, _ = self._get_elo(p1_id, surface)
            p2_final, _, _ = self._get_elo(p2_id, surface)
            
            # Lưu vào list để thêm vào DataFrame
            p1_elos.append(p1_final)
            p2_elos.append(p2_final)
            elo_diffs.append(p1_final - p2_final)
            
            # Cập nhật Elo SAU trận đấu (cho trận tiếp theo)
            if target == 1:
                self._update_elo(p1_id, p2_id, surface)
            else:
                self._update_elo(p2_id, p1_id, surface)
                
        # Thêm cột vào DataFrame
        df_new = df.copy()
        df_new['p1_elo'] = p1_elos
        df_new['p2_elo'] = p2_elos
        df_new['elo_diff'] = elo_diffs
        
        print("Đã tính xong Elo cho toàn bộ lịch sử.")
        return df_new

