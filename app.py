"""
OptiCrop AI – Flask Application
Smart Agricultural Production Optimization Engine
"""

import os, io, json, base64
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, jsonify, send_file, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY']          = 'opticrop-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'sqlite:///' + os.path.join(BASE_DIR, 'database', 'opticrop.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db    = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ── Database Models ────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role     = db.Column(db.String(20), default='farmer')
    created  = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True,
                                  cascade='all, delete-orphan')

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    nitrogen    = db.Column(db.Float)
    phosphorous = db.Column(db.Float)
    potassium   = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity    = db.Column(db.Float)
    ph          = db.Column(db.Float)
    rainfall    = db.Column(db.Float)
    prediction  = db.Column(db.String(50))
    confidence  = db.Column(db.Float)
    date        = db.Column(db.DateTime, default=datetime.utcnow)

class Contact(db.Model):
    __tablename__ = 'contacts'
    id      = db.Column(db.Integer, primary_key=True)
    name    = db.Column(db.String(100))
    email   = db.Column(db.String(120))
    subject = db.Column(db.String(200))
    message = db.Column(db.Text)
    date    = db.Column(db.DateTime, default=datetime.utcnow)

# ── Load ML Model ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def load_model():
    try:
        model   = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
        scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return model, scaler, encoder
    except Exception:
        return None, None, None

ml_model, ml_scaler, ml_encoder = load_model()

# ── Crop Knowledge Base ────────────────────────────────────────────────────────
CROP_INFO = {
    'rice':        {'season':'Kharif',  'temp':'20-35°C', 'humidity':'80-85%', 'ph':'5.5-7.0', 'water':'1200mm', 'fertilizer':'Urea + DAP',       'emoji':'🌾'},
    'maize':       {'season':'Kharif',  'temp':'18-27°C', 'humidity':'55-75%', 'ph':'5.8-7.0', 'water':'500mm',  'fertilizer':'NPK 10-26-26',      'emoji':'🌽'},
    'chickpea':    {'season':'Rabi',    'temp':'10-25°C', 'humidity':'40-60%', 'ph':'6.0-7.5', 'water':'400mm',  'fertilizer':'SSP + Rhizobium',   'emoji':'🫘'},
    'kidneybeans': {'season':'Kharif',  'temp':'18-24°C', 'humidity':'50-70%', 'ph':'6.0-7.5', 'water':'300mm',  'fertilizer':'DAP + MOP',         'emoji':'🫘'},
    'pigeonpeas':  {'season':'Kharif',  'temp':'18-29°C', 'humidity':'60-75%', 'ph':'5.5-7.0', 'water':'650mm',  'fertilizer':'SSP + Rhizobium',   'emoji':'🫘'},
    'mothbeans':   {'season':'Kharif',  'temp':'25-35°C', 'humidity':'40-60%', 'ph':'6.0-7.5', 'water':'200mm',  'fertilizer':'SSP + MOP',         'emoji':'🫘'},
    'mungbean':    {'season':'Kharif',  'temp':'25-35°C', 'humidity':'60-75%', 'ph':'6.2-7.2', 'water':'350mm',  'fertilizer':'DAP + Rhizobium',   'emoji':'🫘'},
    'blackgram':   {'season':'Kharif',  'temp':'25-35°C', 'humidity':'60-80%', 'ph':'6.0-7.0', 'water':'400mm',  'fertilizer':'SSP + Rhizobium',   'emoji':'🫘'},
    'lentil':      {'season':'Rabi',    'temp':'15-25°C', 'humidity':'50-70%', 'ph':'6.0-8.0', 'water':'250mm',  'fertilizer':'SSP + Rhizobium',   'emoji':'🫘'},
    'pomegranate': {'season':'Annual',  'temp':'25-35°C', 'humidity':'50-70%', 'ph':'5.5-7.2', 'water':'500mm',  'fertilizer':'NPK 12-12-17',      'emoji':'🍎'},
    'banana':      {'season':'Annual',  'temp':'20-35°C', 'humidity':'75-85%', 'ph':'5.5-7.0', 'water':'1200mm', 'fertilizer':'NPK 8-10-10',       'emoji':'🍌'},
    'mango':       {'season':'Summer',  'temp':'24-30°C', 'humidity':'50-60%', 'ph':'5.5-7.5', 'water':'750mm',  'fertilizer':'NPK 10-10-10',      'emoji':'🥭'},
    'grapes':      {'season':'Annual',  'temp':'15-35°C', 'humidity':'60-70%', 'ph':'5.5-6.5', 'water':'700mm',  'fertilizer':'NPK 12-12-17',      'emoji':'🍇'},
    'watermelon':  {'season':'Summer',  'temp':'22-30°C', 'humidity':'60-80%', 'ph':'6.0-7.0', 'water':'400mm',  'fertilizer':'NPK 5-10-10',       'emoji':'🍉'},
    'muskmelon':   {'season':'Summer',  'temp':'25-35°C', 'humidity':'60-75%', 'ph':'6.0-7.0', 'water':'350mm',  'fertilizer':'NPK 5-10-10',       'emoji':'🍈'},
    'apple':       {'season':'Winter',  'temp':'10-25°C', 'humidity':'60-70%', 'ph':'5.5-6.5', 'water':'1000mm', 'fertilizer':'NPK 10-10-10',      'emoji':'🍏'},
    'orange':      {'season':'Winter',  'temp':'15-30°C', 'humidity':'60-75%', 'ph':'5.5-7.0', 'water':'900mm',  'fertilizer':'NPK 8-8-8',         'emoji':'🍊'},
    'papaya':      {'season':'Annual',  'temp':'22-30°C', 'humidity':'60-80%', 'ph':'6.0-7.0', 'water':'1000mm', 'fertilizer':'NPK 10-10-10',      'emoji':'🍑'},
    'coconut':     {'season':'Annual',  'temp':'20-32°C', 'humidity':'70-80%', 'ph':'5.5-8.0', 'water':'1500mm', 'fertilizer':'NPK 13-0-45',       'emoji':'🥥'},
    'cotton':      {'season':'Kharif',  'temp':'21-30°C', 'humidity':'50-80%', 'ph':'5.8-8.0', 'water':'700mm',  'fertilizer':'NPK 20-10-10',      'emoji':'🌿'},
    'jute':        {'season':'Kharif',  'temp':'24-37°C', 'humidity':'70-90%', 'ph':'6.0-7.0', 'water':'1500mm', 'fertilizer':'Urea + SSP',        'emoji':'🌿'},
    'coffee':      {'season':'Annual',  'temp':'15-28°C', 'humidity':'70-80%', 'ph':'6.0-6.5', 'water':'1500mm', 'fertilizer':'NPK 17-17-17',      'emoji':'☕'},
}

def get_crop_info(crop_name):
    return CROP_INFO.get(crop_name.lower(), {
        'season':'Annual', 'temp':'20-30°C', 'humidity':'60-80%',
        'ph':'6.0-7.0', 'water':'800mm', 'fertilizer':'NPK Balanced', 'emoji':'🌱'
    })

# ── Auth Decorators ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    total_predictions = Prediction.query.count()
    total_users       = User.query.count()
    return render_template('index.html',
                           total_predictions=total_predictions,
                           total_users=total_users)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        c = Contact(
            name    = request.form.get('name'),
            email   = request.form.get('email'),
            subject = request.form.get('subject'),
            message = request.form.get('message'),
        )
        db.session.add(c)
        db.session.commit()
        flash('Message sent successfully! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not all([name, email, password, confirm]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user   = User(name=name, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id']   = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('admin_dashboard') if user.role == 'admin'
                            else url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

# ── User Dashboard ─────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    user  = User.query.get(session['user_id'])
    preds = Prediction.query.filter_by(user_id=user.id)\
                            .order_by(Prediction.date.desc()).limit(10).all()
    total = Prediction.query.filter_by(user_id=user.id).count()

    # Chart data
    all_preds = Prediction.query.filter_by(user_id=user.id).all()
    crop_counts = {}
    for p in all_preds:
        crop_counts[p.prediction] = crop_counts.get(p.prediction, 0) + 1

    return render_template('dashboard.html', user=user, predictions=preds,
                           total=total, crop_counts=json.dumps(crop_counts))

# ── Prediction ─────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            vals = [
                float(request.form['nitrogen']),
                float(request.form['phosphorous']),
                float(request.form['potassium']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall']),
            ]
        except (ValueError, KeyError):
            flash('Please enter valid numeric values.', 'danger')
            return render_template('predict.html')

        if ml_model is None:
            flash('ML model not loaded. Please run train_model.py first.', 'danger')
            return render_template('predict.html')

        arr     = np.array([vals])
        scaled  = ml_scaler.transform(arr)
        pred_idx = ml_model.predict(scaled)[0]
        proba   = ml_model.predict_proba(scaled)[0]
        confidence = round(float(np.max(proba)) * 100, 2)
        crop_name  = ml_encoder.inverse_transform([pred_idx])[0]

        # Save to DB
        p = Prediction(
            user_id=session['user_id'],
            nitrogen=vals[0], phosphorous=vals[1], potassium=vals[2],
            temperature=vals[3], humidity=vals[4], ph=vals[5], rainfall=vals[6],
            prediction=crop_name, confidence=confidence,
        )
        db.session.add(p)
        db.session.commit()

        info = get_crop_info(crop_name)
        return render_template('result.html',
                               crop=crop_name, confidence=confidence,
                               info=info, pred_id=p.id,
                               inputs=dict(zip(
                                   ['Nitrogen','Phosphorous','Potassium',
                                    'Temperature','Humidity','pH','Rainfall'], vals)))
    return render_template('predict.html')

# ── History ────────────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    page  = request.args.get('page', 1, type=int)
    query = request.args.get('q', '')
    base  = Prediction.query.filter_by(user_id=session['user_id'])
    if query:
        base = base.filter(Prediction.prediction.ilike(f'%{query}%'))
    preds = base.order_by(Prediction.date.desc()).paginate(page=page, per_page=10)
    return render_template('history.html', predictions=preds, query=query)

# ── PDF Report ─────────────────────────────────────────────────────────────────
@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    p    = Prediction.query.get_or_404(pred_id)
    user = User.query.get(session['user_id'])
    if p.user_id != user.id and user.role != 'admin':
        abort(403)

    info = get_crop_info(p.prediction)
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              rightMargin=inch*0.75, leftMargin=inch*0.75,
                              topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Title'],
                                 textColor=colors.HexColor('#2d6a4f'), fontSize=22)
    h2_style    = ParagraphStyle('h2', parent=styles['Heading2'],
                                 textColor=colors.HexColor('#40916c'))
    story = [
        Paragraph("🌿 OptiCrop AI – Crop Recommendation Report", title_style),
        Spacer(1, 0.2*inch),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles['Normal']),
        Paragraph(f"Farmer: {user.name}  |  Email: {user.email}", styles['Normal']),
        Spacer(1, 0.3*inch),
        Paragraph("Recommended Crop", h2_style),
        Paragraph(f"<b>{p.prediction.upper()}</b>  –  Confidence: {p.confidence:.1f}%", styles['Normal']),
        Spacer(1, 0.2*inch),
        Paragraph("Soil Input Parameters", h2_style),
    ]

    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Nitrogen (N)',    p.nitrogen,    'mg/kg'],
        ['Phosphorous (P)', p.phosphorous, 'mg/kg'],
        ['Potassium (K)',   p.potassium,   'mg/kg'],
        ['Temperature',     p.temperature, '°C'],
        ['Humidity',        p.humidity,    '%'],
        ['Soil pH',         p.ph,          ''],
        ['Rainfall',        p.rainfall,    'mm'],
    ]
    t = Table(input_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d6a4f')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0fdf4'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story += [t, Spacer(1, 0.3*inch), Paragraph("Crop Details", h2_style)]

    crop_data = [
        ['Property', 'Value'],
        ['Season',              info['season']],
        ['Ideal Temperature',   info['temp']],
        ['Ideal Humidity',      info['humidity']],
        ['Ideal pH',            info['ph']],
        ['Water Requirement',   info['water']],
        ['Recommended Fertilizer', info['fertilizer']],
    ]
    t2 = Table(crop_data, colWidths=[3*inch, 3*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#40916c')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0fdf4'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
    ]))
    story += [t2, Spacer(1, 0.4*inch),
              Paragraph("Disclaimer: This recommendation is AI-generated. "
                        "Consult an agronomist for final decisions.", styles['Italic'])]

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f'OptiCrop_Report_{p.prediction}_{pred_id}.pdf',
                     mimetype='application/pdf')

# ── Admin ──────────────────────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_dashboard():
    users        = User.query.order_by(User.created.desc()).all()
    predictions  = Prediction.query.order_by(Prediction.date.desc()).limit(20).all()
    total_users  = User.query.count()
    total_preds  = Prediction.query.count()
    contacts     = Contact.query.order_by(Contact.date.desc()).all()

    # Crop frequency for chart
    all_preds = Prediction.query.all()
    crop_freq = {}
    for p in all_preds:
        crop_freq[p.prediction] = crop_freq.get(p.prediction, 0) + 1

    # Load model metrics
    metrics_path = os.path.join(BASE_DIR, 'model', 'metrics.csv')
    metrics = []
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path).to_dict('records')

    return render_template('admin.html',
                           users=users, predictions=predictions,
                           total_users=total_users, total_preds=total_preds,
                           contacts=contacts, crop_freq=json.dumps(crop_freq),
                           metrics=metrics)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot delete admin account.', 'danger')
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.name} deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

# ── API ────────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    try:
        vals = [float(data[k]) for k in
                ['nitrogen','phosphorous','potassium','temperature','humidity','ph','rainfall']]
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    if ml_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    scaled    = ml_scaler.transform([vals])
    pred_idx  = ml_model.predict(scaled)[0]
    proba     = ml_model.predict_proba(scaled)[0]
    crop_name = ml_encoder.inverse_transform([pred_idx])[0]
    confidence = round(float(np.max(proba)) * 100, 2)
    info = get_crop_info(crop_name)
    return jsonify({'crop': crop_name, 'confidence': confidence, 'info': info})

# ── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(403)
def forbidden(e):
    return render_template('404.html', code=403,
                           message="Access Forbidden"), 403

# ── Init DB & Seed Admin ───────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role='admin').first():
            admin = User(
                name     = 'Admin',
                email    = 'admin@opticrop.ai',
                password = bcrypt.generate_password_hash('admin123').decode('utf-8'),
                role     = 'admin',
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin seeded: admin@opticrop.ai / admin123")

# ── Jinja Globals & Filters ───────────────────────────────────────────────────
import json as _json

@app.context_processor
def inject_globals():
    return {'now': datetime.utcnow}

@app.template_filter('fromjson')
def fromjson_filter(s):
    try:
        return _json.loads(s)
    except Exception:
        return {}

# Initialize DB on startup (works for both local and Render)
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
