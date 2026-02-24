<?php
/**
 * Microcenter.gr Per-Store Stock API
 *
 * INSTALLATION:
 *   Upload this file to: /catalog/controller/api/stock_locations.php
 *   on your OpenCart server (www.microcenter.gr)
 *
 * USAGE:
 *   GET /index.php?route=api/stock_locations&api_token=TOKEN&search=QUERY
 *
 * RETURNS:
 *   {
 *     "status": "success",
 *     "products": [
 *       {
 *         "product_id": "123",
 *         "name": "...",
 *         "price": "5,99€",
 *         "qty_store":  1,   <- Αμπελόκηποι (Ποσότητα καταστήματος)
 *         "qty_branch": 0,   <- Παγκράτι   (Ποσότητα υποκαταστήματος)
 *         "href": "https://..."
 *       }
 *     ]
 *   }
 */
class ControllerApiStockLocations extends Controller {

    // Adjust these if your column names differ
    const COL_STORE  = 'quantity';         // Ποσότητα καταστήματος  (Αμπελόκηποι)
    const COL_BRANCH = 'quantity2';        // Ποσότητα υποκαταστήματος (Παγκράτι)

    public function index() {
        $this->response->addHeader('Content-Type: application/json');

        // Require valid API session
        if (empty($this->session->data['api_id'])) {
            $this->response->setOutput(json_encode(['error' => 'Unauthorized']));
            return;
        }

        $search = html_entity_decode(
            isset($this->request->get['search']) ? $this->request->get['search'] : '',
            ENT_QUOTES, 'UTF-8'
        );

        if (!$search) {
            $this->response->setOutput(json_encode(['error' => 'Missing search parameter']));
            return;
        }

        // Detect which branch column exists in oc_product
        $branch_col = $this->detectBranchColumn();

        // Search products
        $sql = "
            SELECT
                p.product_id,
                pd.name,
                p.price,
                p." . self::COL_STORE . " AS qty_store,
                " . ($branch_col ? "p.`$branch_col` AS qty_branch" : "0 AS qty_branch") . ",
                p.model
            FROM `" . DB_PREFIX . "product` p
            LEFT JOIN `" . DB_PREFIX . "product_description` pd
                ON (p.product_id = pd.product_id AND pd.language_id = '" . (int)$this->config->get('config_language_id') . "')
            WHERE pd.name LIKE '%" . $this->db->escape($search) . "%'
            AND p.status = 1
            LIMIT 10
        ";

        $result = $this->db->query($sql);
        $products = [];

        foreach ($result->rows as $row) {
            $products[] = [
                'product_id' => $row['product_id'],
                'name'       => $row['name'],
                'price'      => $this->currency->format(
                    $row['price'],
                    $this->config->get('config_currency')
                ),
                'qty_store'  => (int)$row['qty_store'],
                'qty_branch' => (int)$row['qty_branch'],
                'model'      => $row['model'],
                'href'       => $this->url->link(
                    'product/product',
                    'product_id=' . $row['product_id'],
                    true
                ),
            ];
        }

        $this->response->setOutput(json_encode([
            'status'       => 'success',
            'branch_col'   => $branch_col ?: 'not found',
            'products'     => $products,
        ]));
    }

    /**
     * Auto-detect the branch quantity column name.
     * Returns the column name string, or null if not found.
     */
    private function detectBranchColumn() {
        $candidates = [
            'quantity2',
            'quantity_branch',
            'quantity_sub',
            'sub_quantity',
            'branch_quantity',
            'qty_branch',
        ];

        $cols = $this->db->query("SHOW COLUMNS FROM `" . DB_PREFIX . "product`");
        $existing = array_column($cols->rows, 'Field');

        foreach ($candidates as $col) {
            if (in_array($col, $existing)) {
                return $col;
            }
        }
        return null;
    }
}
