<?php
/**
 * microcenter.gr — Order Status API για το chatbot
 *
 * ΕΓΚΑΤΑΣΤΑΣΗ:
 *   Ανέβασε αυτό το αρχείο στο root του OpenCart σου:
 *   π.χ. https://www.microcenter.gr/order_status.php
 *
 * ΑΣΦΑΛΕΙΑ:
 *   Άλλαξε το ORDER_STATUS_KEY παρακάτω σε κάτι τυχαίο.
 *   Βάλε ΤΟ ΙΔΙΟ key και στο Render ως env var ORDER_STATUS_KEY.
 *
 * ΧΡΗΣΗ:
 *   GET https://www.microcenter.gr/order_status.php?key=SECRET&order_id=511488
 */

define('ORDER_STATUS_KEY', 'CHANGE_THIS_SECRET_KEY');

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');

// --- Auth ---
if (empty($_GET['key']) || $_GET['key'] !== ORDER_STATUS_KEY) {
    http_response_code(403);
    echo json_encode(['error' => 'unauthorized']);
    exit;
}

// --- Validate order_id ---
$order_id = isset($_GET['order_id']) ? (int)$_GET['order_id'] : 0;
if ($order_id <= 0) {
    http_response_code(400);
    echo json_encode(['error' => 'invalid order_id']);
    exit;
}

// --- Load OpenCart DB config ---
$config_file = __DIR__ . '/config.php';
if (!file_exists($config_file)) {
    http_response_code(500);
    echo json_encode(['error' => 'config.php not found']);
    exit;
}
require_once $config_file;

// --- Connect ---
$conn = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(['error' => 'db_connection_failed']);
    exit;
}
$conn->set_charset('utf8');

// --- Find Greek language_id ---
$lang_id = 1;
$lr = $conn->query("SELECT language_id FROM `" . DB_PREFIX . "language` WHERE `code` = 'el' LIMIT 1");
if ($lr && $row = $lr->fetch_assoc()) {
    $lang_id = (int)$row['language_id'];
}

// --- Query order ---
$stmt = $conn->prepare(
    "SELECT
        o.order_id,
        DATE_FORMAT(o.date_added, '%d/%m/%Y') AS date_added,
        CONCAT(FORMAT(o.total, 2), ' €')       AS total,
        COALESCE(o.tracking, '')               AS tracking,
        COALESCE(os.name, '')                  AS order_status
     FROM `" . DB_PREFIX . "order` o
     LEFT JOIN `" . DB_PREFIX . "order_status` os
           ON (o.order_status_id = os.order_status_id AND os.language_id = ?)
     WHERE o.order_id = ?
     LIMIT 1"
);
$stmt->bind_param('ii', $lang_id, $order_id);
$stmt->execute();
$order = $stmt->get_result()->fetch_assoc();
$stmt->close();
$conn->close();

echo json_encode($order ?: ['order_id' => null]);
