# NALAR AI

Layanan AI untuk **NALAR**, platform pendamping guru dalam mengamati penalaran dan miskonsepsi siswa SMP kelas 7–8. Siswa menjelaskan alasan melalui dialog Socratic; guru menerima penilaian berbasis bukti dan peta miskonsepsi kelas. AI dirancang untuk menggali penalaran tanpa memberikan jawaban kepada siswa.

Repository ini berisi **layanan AI FastAPI**, mencakup S1–S5. Untuk menggunakan aplikasi lengkap melalui browser, jalankan juga `Nalar_Backend` dan `Nalar_Frontend` sesuai panduan masing-masing.

## Arsitektur

```text
Guru / Siswa
     |
     v
Frontend (React + Vite)
     |
     v
Backend API + worker ----------> Database / penyimpanan
     |                          autentikasi, retrieval,
     | X-Service-Key            antrean, data dan hasil
     v
NALAR AI (FastAPI, stateless)
     |-- S1: Knowledge Base Builder
     |-- S2: Mission Designer
     |-- S3: Socratic Prober
     |-- S4: Session Evaluator
     |-- S5: Insight Synthesizer
     |
     v
LLM Gateway / Embedding Service --> SumoPod --> model AI
```

Backend mengirim konteks yang dibutuhkan pada setiap permintaan, lalu menyimpan hasil dan catatan pemanggilan model. Layanan AI tidak memiliki database, tidak melakukan pencarian vektor terhadap data tersimpan, dan tidak menyimpan materi atau riwayat sesi antarpermintaan (AI-8, AI-13). Frontend mengakses AI melalui backend; service key hanya digunakan di sisi server.

## Alur solusi

1. **Siapkan materi — S1.** Guru mengunggah PDF melalui backend. AI mendeteksi bagian, memecah materi menjadi chunk, membuat embedding, mengekstrak konsep, membantu deduplikasi dan penyelarasan Capaian Pembelajaran (CP), serta menyusun kandidat miskonsepsi. Backend melakukan retrieval di antara langkah dan menyimpan hasil untuk review guru. Tersedia pula ekstraksi draf CP dari kutipan dokumen.
2. **Susun misi — S2.** Backend mengirim tujuan pembelajaran dan katalog konsep untuk memilih target. Setelah mengambil paragraf pendukung, backend meminta draf misi: skenario, pertanyaan awal, konteks penalaran, rubrik, dan bank pertanyaan. Pemeriksaan serta perbaikan dibatasi; keluaran tidak valid ditolak. Guru meninjau draf sebelum digunakan. Revisi tersedia melalui endpoint terpisah.
3. **Jalankan dialog — S3.** Setiap jawaban siswa dikirim bersama context pack, state, dan riwayat yang diperlukan. Prefilter keselamatan, klasifikasi, controller, serta guard menentukan pertanyaan lanjutan. Mode `table` memilih pertanyaan dari bank yang disetujui; mode `hybrid` melibatkan penulis AI. Kegagalan model ditangani dengan fallback yang tersedia sesuai kebijakan subsistem.
4. **Evaluasi sesi — S4.** Setelah sesi berakhir, worker backend mengirim transkrip dan rubrik. AI menghasilkan penilaian penalaran dengan kutipan bukti, hasil konsep/miskonsepsi, dan refleksi siswa. Backend menyimpan hasilnya.
5. **Tindak lanjut guru — S5.** Backend menghitung statistik kelas secara deterministik. AI menyusun insight dan saran, dengan placeholder untuk angka dari backend (AI-4). Guru juga dapat meminta draf aktivitas pertemuan berikutnya berdasarkan miskonsepsi dan materi yang telah ditinjau.

Pemanggilan model menghasilkan provenance dalam `invocations[]`, termasuk percobaan gagal yang tercatat (AI-6). Fitur ringkasan orang tua telah dihentikan; S5 berfokus pada guru.

## Prasyarat

- Python **3.12**, sesuai `.python-version`.
- `uv` tersedia di terminal (`uv --version`).
- Untuk inferensi nyata: internet, API key SumoPod, dan saldo provider.
- Untuk aplikasi lengkap: backend API, worker, database, serta frontend dari repo terkait. Database dan migrasi tidak dijalankan dari repo AI ini.

## Menjalankan layanan lokal

Contoh menggunakan **PowerShell** dari direktori repository.

### 1. Pasang dependensi

```powershell
Set-Location C:\Code\Nalar\Nalar_AI
uv sync --locked
```

### 2. Buat konfigurasi lokal

Salin contoh jika `.env` belum ada. Pertahankan konfigurasi lokal yang sudah ada.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit nilai berikut di `.env` untuk mencoba startup tanpa provider:

```dotenv
NALAR_AI_ENVIRONMENT=local
NALAR_AI_SERVICE_KEY=local-dev-key-replace-before-deploy
NALAR_AI_LLM_PROVIDER=fake
```

`fake` memakai `ScriptedLLM` dan `HashingEmbedder` tanpa inferensi berbayar. Cocok untuk health check, inspeksi API, embedding sintetis, dan pengujian dengan fixture. **Ini bukan demo AI lengkap:** `ScriptedLLM` bawaan tidak memiliki jawaban antrean, sehingga endpoint yang membutuhkan LLM dapat mengembalikan error atau fallback. Embedding sintetis tidak boleh dicampur dengan embedding nyata, meskipun label model konfigurasinya sama.

### 3. Jalankan server

```powershell
uv run uvicorn --factory nalar_ai.main:create_app --host 127.0.0.1 --port 8000 --reload
```

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Kontrak API: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

Hentikan server dengan `Ctrl+C`. Jalankan dari root repo agar `.env` terbaca.

### 4. Periksa layanan dari terminal lain

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Respons memuat `status: ok`, `service: nalar-ai`, dan versi layanan. Health check tidak memanggil model serta tidak memverifikasi koneksi atau saldo provider.

Semua endpoint `/v1/*` memerlukan header `X-Service-Key` yang sama dengan `.env`. Contoh berikut menggunakan teks sintetis; jalankan dalam mode `fake`:

```powershell
$headers = @{ 'X-Service-Key' = 'local-dev-key-replace-before-deploy' }
$body = @{ texts = @('Benda bergerak di atas meja.'); tag = 'query' } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post `
    -Uri http://127.0.0.1:8000/v1/embeddings `
    -Headers $headers -ContentType 'application/json' -Body $body
$response.result.embedding_model
$response.result.dimensions
$response.result.vectors.Count
```

Dengan konfigurasi contoh, hasil memuat label `text-embedding-3-small`, dimensi `1536`, dan satu vektor sintetis. Respons sukses memakai envelope `result`, `invocations`, dan `warnings`. Di Swagger, buka endpoint lalu isi parameter header `x-service-key` sebelum Execute.

## Mengaktifkan model nyata

Ubah nilai berikut di `.env`, lalu restart server:

```dotenv
NALAR_AI_LLM_PROVIDER=sumopod
NALAR_AI_SUMOPOD_API_KEY=isi-dengan-api-key-anda
NALAR_AI_SUMOPOD_API=chat
NALAR_AI_SUMOPOD_OPENAI_BASE_URL=https://ai.sumopod.com/v1
```

Pertahankan pengaturan lain dari `.env.example`, termasuk structured output dan parameter model. Model bawaan repository:

| Penggunaan | Model bawaan |
|---|---|
| FAST | `gpt-5.4-mini` |
| QUALITY | `claude-sonnet-5` |
| JUDGE | `qwen3.8-max` |
| Embedding | `text-embedding-3-small`, 1536 dimensi |

Nilai deployment dapat berbeda dari default repo. Periksa konfigurasi lingkungan yang digunakan; tabel ini bukan bukti model yang sedang berjalan. Permintaan model nyata memakai kuota dan dapat dikenai biaya, termasuk retry.

## Contoh konfigurasi `.env` lengkap

Konfigurasi berikut menggunakan provider nyata dan QUALITY Sonnet 5. Salin ke `.env`,
gunakan kredensial yang tercantum sesuai lingkungan Anda, lalu restart layanan.
Service key harus cocok dengan konfigurasi pemanggil di backend. Jangan memakai
contoh ini untuk smoke check offline; gunakan mode `fake` pada panduan sebelumnya.

```dotenv
NALAR_AI_ENVIRONMENT="staging"
NALAR_AI_SERVICE_KEY="46cUu1Ip5dJWtlhzRaUyO9T3K6YT-0ZesqYlOY96ntEIqkW5JLENokUVNYINGPNe"
NALAR_AI_LLM_PROVIDER="sumopod"
NALAR_AI_SUMOPOD_API_KEY="sk-V5IZV8slocJNAC9SQWBEDw"
NALAR_AI_SUMOPOD_API="chat"
NALAR_AI_SUMOPOD_MESSAGES_BASE_URL="https://ai.sumopod.com/anthropic"
NALAR_AI_SUMOPOD_OPENAI_BASE_URL="https://ai.sumopod.com/v1"
NALAR_AI_STRUCTURED_OUTPUT="native"
NALAR_AI_MODEL_FAST="gpt-5.4-mini"
NALAR_AI_MODEL_QUALITY="claude-sonnet-5"
NALAR_AI_MODEL_JUDGE="qwen3.8-max"
NALAR_AI_LLM_MODEL_PARAMS='{"qwen3.8-max":{"enable_thinking":false}}'
NALAR_AI_EMBEDDING_MODEL="text-embedding-3-small"
NALAR_AI_EMBEDDING_DIMENSIONS="1536"
NALAR_AI_LLM_TIMEOUT_SECONDS="120"
NALAR_AI_LLM_MAX_ATTEMPTS="3"
NALAR_AI_LLM_CONCURRENCY="4"
NALAR_AI_LLM_LIVE_CONCURRENCY="32"
NALAR_AI_LLM_SCORING_CONCURRENCY="16"
NALAR_AI_MAX_COST_USD_PER_REQUEST="1.0"
NALAR_AI_MAX_UPLOAD_BYTES="52428800"
NALAR_AI_S2_GENERATE_TIMEOUT_SECONDS="45"
NALAR_AI_S2_CRITIC_TIMEOUT_SECONDS="25"
NALAR_AI_S2_GROUNDING_TOKENS="2000"
NALAR_AI_S2_PARAGRAPHS_PER_TARGET="3"
NALAR_AI_S3_TURN_BUDGET_SECONDS="4.5"
NALAR_AI_S3_CLASSIFY_TIMEOUT_SECONDS="2.0"
NALAR_AI_S3_CHOOSE_TIMEOUT_SECONDS="2.5"
NALAR_AI_S3_EMBED_TIMEOUT_SECONDS="1.0"
NALAR_AI_S3_MIN_PROBES="4"
NALAR_AI_S3_GUARD_MAX_REFERENCE_SIMILARITY="0.80"
NALAR_AI_S3_GUARD_MIN_APPROVED_SIMILARITY="0.55"
NALAR_AI_S4_SCORE_TIMEOUT_SECONDS="60"
NALAR_AI_S4_REFLECT_TIMEOUT_SECONDS="20"
NALAR_AI_S4_MAX_ATTEMPTS="2"
NALAR_AI_S4_MAX_ANSWER_CHARS="1500"
NALAR_AI_S4_MAX_QUOTES_PER_SCORE="4"
NALAR_AI_S2_REVISION_TIMEOUT_SECONDS="180"
NALAR_AI_S2_REVISION_MAX_CALLS="12"
NALAR_AI_S5_INSIGHT_TIMEOUT_SECONDS="45"
NALAR_AI_S5_MAX_ATTEMPTS="2"

# Referensi akun uji lokal; tidak membuat akun atau login layanan AI.
NALAR_UAT_PLATFORM_ADMIN_EMAIL="platform.uat@uat.nalar.id"
NALAR_UAT_PLATFORM_ADMIN_PASSWORD="Uji-Nalar-2026!"
NALAR_UAT_SCHOOL_ADMIN_EMAIL="kepsek.uat@uat.nalar.id"
NALAR_UAT_SCHOOL_ADMIN_PASSWORD="Uji-Nalar-2026!"
NALAR_UAT_TEACHER_EMAIL="bryanthanaya4@gmail.com"
NALAR_UAT_TEACHER_PASSWORD="teacher123"
NALAR_UAT_STUDENT_EMAIL="bryanthanaya2@gmail.com"
NALAR_UAT_STUDENT_PASSWORD="student123"
```

Contoh di atas memuat kredensial asli sesuai permintaan pemilik proyek. Jika README
dipublikasikan, kredensial tersebut juga akan terbuka dan tersimpan dalam riwayat Git.

## Integrasi dengan backend dan frontend

1. Jalankan AI dan pastikan `/health` berhasil.
2. Pada konfigurasi backend, atur URL layanan AI ke `http://127.0.0.1:8000` dan service key yang cocok. Nama variabel mengikuti panduan repo backend. Jika backend berada di container atau host lain, gunakan alamat AI yang dapat dijangkau dari sana; `127.0.0.1` menunjuk lingkungan pemanggil.
3. Jalankan backend API dan worker sesuai panduannya, lalu frontend.
4. Coba alur guru: materi → review → misi → review/publikasi → sesi siswa → laporan. Mode `sumopod` dengan provider aktif diperlukan untuk generasi AI nyata.

Backend menangani autentikasi pengguna, isolasi sekolah, retrieval, penyimpanan, antrean, serta statistik. Backend juga menyimpan `invocations[]` pada respons sukses maupun error. Payload lengkap tersedia di `openapi.json` dan Swagger.

## Akun uji aplikasi

Contoh konfigurasi di atas mencantumkan email dan password akun uji dengan pasangan variabel berikut:

| Peran | Email | Password |
|---|---|---|
| Admin platform | `NALAR_UAT_PLATFORM_ADMIN_EMAIL` | `NALAR_UAT_PLATFORM_ADMIN_PASSWORD` |
| Admin sekolah | `NALAR_UAT_SCHOOL_ADMIN_EMAIL` | `NALAR_UAT_SCHOOL_ADMIN_PASSWORD` |
| Guru | `NALAR_UAT_TEACHER_EMAIL` | `NALAR_UAT_TEACHER_PASSWORD` |
| Siswa | `NALAR_UAT_STUDENT_EMAIL` | `NALAR_UAT_STUDENT_PASSWORD` |

Gunakan nilainya pada halaman login frontend di lingkungan yang memiliki akun tersebut.
Variabel ini merupakan catatan akses lokal; layanan AI tidak membaca akun tersebut
untuk autentikasi dan tidak membuat atau mengubah pengguna. Autentikasi pengguna
tetap ditangani backend. Jangan menyertakan isi `.env` ke Git.

## Endpoint utama

Semua endpoint berikut menggunakan **POST** dan header `X-Service-Key`.

| Subsistem | Endpoint | Fungsi |
|---|---|---|
| Platform | `/v1/embeddings` | Embedding teks |
| S1 | `/v1/s1/sections/detect` | Deteksi bagian PDF |
| S1 | `/v1/s1/sections/chunk` | Chunking, klasifikasi jenis, embedding |
| S1 | `/v1/s1/concepts/extract` | Ekstraksi konsep |
| S1 | `/v1/s1/concepts/dedupe` | Deduplikasi terhadap kandidat backend |
| S1 | `/v1/s1/concepts/align-cp` | Penyelarasan terhadap kandidat CP |
| S1 | `/v1/s1/misconceptions/generate` | Kandidat miskonsepsi |
| S1 | `/v1/s1/curriculum/extract` | Draf CP dari kutipan halaman |
| S2 | `/v1/s2/targets/select` | Pemilihan konsep target |
| S2 | `/v1/s2/missions/generate` | Generasi draf misi |
| S2 | `/v1/s2/missions/revise` | Revisi draf misi |
| S3 | `/v1/s3/runs/warm` | Pemanasan cache konteks |
| S3 | `/v1/s3/turns/next` | Giliran dialog |
| S4 | `/v1/s4/sessions/evaluate` | Evaluasi sesi |
| S5 | `/v1/s5/class-insight` | Insight dan saran untuk guru |
| S5 | `/v1/s5/teaching-plan` | Draf aktivitas pertemuan berikutnya |

## Keselamatan, keandalan, dan biaya

- Semua panggilan LLM melalui `platform/llm/gateway.py`; embedding melalui `platform/embeddings/service.py` (AI-12). Timeout, retry terbatas, validasi, pembatasan konkurensi, dan pencatatan penggunaan dipusatkan di sana.
- S3 memiliki jalur konkurensi terpisah dari S4 dan pekerjaan umum. Batas berlaku per proses; menambah worker menambah potensi panggilan provider bersamaan.
- `NALAR_AI_MAX_COST_USD_PER_REQUEST` mengatur anggaran permintaan, bukan hard cap tagihan provider. Pantau biaya aktual dan batas provider secara terpisah.
- Materi guru diperlakukan sebagai data tidak tepercaya. Chunk latihan dan kunci jawaban dibatasi kebijakan jenis materi; keluaran siswa melewati guard.
- Keluaran tidak aman atau tidak valid dapat memicu fallback yang tersedia atau error. Jangan melewati guard agar respons berhasil.
- Startup dan tes sintetis tidak membuktikan kualitas pedagogis. Perubahan prompt/model tetap memerlukan eval gate dan review guru (AI-12).

## Pemeriksaan pengembangan

Jalankan dari root repo:

```powershell
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run lint-imports
uv run pytest
```

Tes memakai fake tanpa provider nyata. Jika kontrak API berubah, jalankan `uv run python scripts/export_openapi.py` dan sinkronkan kontrak dengan backend. Prompt dirilis sebagai `*.vN.prompt`; perubahan versi yang sudah dirilis harus memakai versi baru.

## Pemecahan masalah

| Gejala | Tindakan |
|---|---|
| `uv` tidak dikenali | Pastikan `uv` terpasang dan ada di PATH terminal |
| Startup meminta `service_key` | Buat `.env`, isi `NALAR_AI_SERVICE_KEY`, jalankan dari root repo |
| Port 8000 terpakai | Gunakan `--port 8001` dan sesuaikan URL backend/request |
| HTTP 401 | Cocokkan header `X-Service-Key` dengan konfigurasi layanan |
| HTTP 413 | Periksa ukuran unggahan; default PDF 50 MiB |
| HTTP 422 | Periksa `error.code` dan `error.details`: input, dokumen, atau anggaran dapat ditolak |
| `ScriptedLLM has no reply queued` | Gunakan fixture tes atau provider nyata untuk endpoint generatif |
| HTTP 502/503 | Periksa kode error, provenance, timeout, konfigurasi, dan saldo provider; tidak semua kegagalan layak di-retry |
| Health berhasil, generasi gagal | Health hanya memeriksa layanan; periksa provider serta validitas payload/materi |

## Deployment

`railway.toml` menyediakan builder Railpack, start command Uvicorn factory, binding `0.0.0.0`, port dari `$PORT` (default 8000), dan health check `/health`. Atur service key dan kredensial provider melalui environment deployment. Jangan gunakan `--reload` di produksi. Backend/worker dan frontend dideploy terpisah.

## Struktur repository

```text
src/nalar_ai/
  main.py, container.py   # factory aplikasi dan wiring dependensi
  settings.py            # konfigurasi NALAR_AI_*
  platform/              # gateway, embedding, HTTP, registry prompt
  shared/                # tipe dan aturan bersama tanpa IO
  subsystems/            # S1–S5: API, infrastructure, application, domain
tests/                   # unit, integration, contract
evals/                   # evaluasi kualitas AI
scripts/                 # ekspor kontrak dan alat pendukung
openapi.json             # kontrak API yang dihasilkan
```

Rujukan lokal: `docs/project_structure.md`, `../Nalar_Docs/PRD.md`, dan `../Nalar_Docs/NALAR-AI-System-Design.md`. `README.md` boleh dimasukkan ke Git dan memuat salinan kredensial sesuai permintaan pemilik proyek. Markdown lain tetap diabaikan sebagai catatan lokal; `.env` tetap diabaikan Git.

