"""
ai_care_assistant.py - NeuroScan AI Patient Care & Educational Health Engine
Provides verified medical information on brain tumors, treatment options,
precautions, emergency red flags, recovery, and questions for physicians.
Sourced from NCI, WHO, American Cancer Society, and NHS UK guidelines.
"""

import os
import re
import json
import logging
import requests

logger = logging.getLogger(__name__)

# Key verified medical source references
MEDICAL_SOURCES = [
    {
        "name": "National Cancer Institute (NCI) - Adult Brain Tumors",
        "url": "https://www.cancer.gov/types/brain",
        "organization": "U.S. National Institutes of Health (NIH)"
    },
    {
        "name": "World Health Organization (WHO) - CNS Tumors Classification",
        "url": "https://www.iarc.who.int/",
        "organization": "World Health Organization"
    },
    {
        "name": "American Cancer Society (ACS) - Brain & Spinal Cord Tumors",
        "url": "https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults.html",
        "organization": "American Cancer Society"
    },
    {
        "name": "National Health Service (NHS) - Brain Tumour Overview",
        "url": "https://www.nhs.uk/conditions/brain-tumours/",
        "organization": "NHS UK"
    },
    {
        "name": "National Brain Tumor Society (NBTS) - Patient & Caregiver Resources",
        "url": "https://braintumor.org/",
        "organization": "National Brain Tumor Society"
    }
]

# Curated Medical Knowledge Base
KNOWLEDGE_BASE = {
    "what_is_tumor": {
        "title": "What is a Brain Tumor?",
        "summary": (
            "A brain tumor is an abnormal proliferation or mass of cells within the brain tissue "
            "or its protective coverings (the meninges). Because the skull is rigid with a fixed volume, "
            "any abnormal growth can increase intracranial pressure and compromise neurological function."
        ),
        "details": [
            "**Primary vs. Metastatic (Secondary):** Primary brain tumors originate inside the brain parenchyma, meninges, or cranial nerves. Metastatic brain tumors begin elsewhere in the body (such as lungs, breasts, skin, or colon) and spread to the brain via the bloodstream.",
            "**Benign vs. Malignant:** Benign tumors (such as many Grade I meningiomas or pituitary adenomas) are slow-growing, well-circumscribed, and do not invade distant tissue. Malignant tumors (such as high-grade gliomas) grow rapidly and diffusely infiltrate surrounding healthy brain parenchyma.",
            "**WHO Grading System (Grades I to IV):** The World Health Organization classifies brain tumors by cellular aggressiveness under a microscope. Grade I tumors are least aggressive and often curable with surgery; Grade IV tumors (e.g., Glioblastoma) are fast-growing, highly vascularized, and require aggressive multimodal therapies."
        ],
        "sources": ["NCI", "WHO CNS Classification", "ACS"]
    },
    "glioma": {
        "title": "Glioma: Overview, Subtypes & Clinical Characteristics",
        "summary": (
            "Gliomas are primary intra-axial tumors originating from the glial support cells "
            "(astrocytes, oligodendrocytes, and ependymal cells) that surround and protect neurons. "
            "They represent approximately 75% of all malignant primary brain tumors in adults."
        ),
        "details": [
            "**Astrocytoma:** Originates from star-shaped astrocytes. Ranges from low-grade (Grade II) to Grade IV (Glioblastoma Multiforme / GBM), which is the most common and aggressive primary brain cancer in adults.",
            "**Oligodendroglioma:** Arises from oligodendrocytes. Characterized by specific genetic biomarkers: IDH mutation and 1p/19q co-deletion, which typically predict better response to chemotherapy and radiation.",
            "**Ependymoma:** Develops from ependymal cells lining the ventricular pathways and central canal of the spinal cord.",
            "**Common Symptoms:** Persistent headaches (frequently worse in the morning), new-onset seizures, focal motor weakness on one side of the body, sensory changes, speech or language difficulty (aphasia), and subtle cognitive or behavioral changes.",
            "**Molecular Biomarkers:** Modern neuro-oncology classifies gliomas using molecular markers including IDH1/IDH2 mutation status, 1p/19q codeletion, and MGMT promoter methylation, which guide targeted therapy selection."
        ],
        "sources": ["WHO 2021 CNS Classification", "NCI", "ACS"]
    },
    "meningioma": {
        "title": "Meningioma: Overview & Characteristics",
        "summary": (
            "Meningiomas arise from the arachnoid cap cells of the meninges—the protective membranes "
            "enveloping the brain and spinal cord. They are the most common primary central nervous system tumor in adults, "
            "accounting for approximately 35–40% of all primary brain neoplasms."
        ),
        "details": [
            "**Benign Nature:** Approximately 80% to 90% of meningiomas are Grade I (benign), characterized by slow expansion and distinct anatomical boundaries (extra-axial).",
            "**Grades II & III:** Grade II (Atypical) represents about 10–15% and has a higher recurrence risk. Grade III (Anaplastic / Malignant) is rare (1–2%) and exhibits aggressive invasive behavior.",
            "**Mechanical Compression:** Because meningiomas grow from the meninges outside the brain tissue, they typically cause symptoms through physical compression of adjacent cerebral cortex or cranial nerves rather than direct cellular infiltration.",
            "**Management Pathways:** Small, asymptomatic, incidental meningiomas found on MRI are often managed through **active surveillance (watchful waiting)** with serial scans every 6 to 12 months. Symptomatic or expanding tumors are treated with surgical resection (craniotomy) or stereotactic radiosurgery (SRS)."
        ],
        "sources": ["NCI", "American Cancer Society", "NHS UK"]
    },
    "pituitary": {
        "title": "Pituitary Tumors (Adenomas): Overview & Characteristics",
        "summary": (
            "Pituitary adenomas originate in the pituitary gland (the body's master endocrine gland) "
            "situated at the base of the skull in a bony cavity called the sella turcica, directly below the optic chiasm. "
            "The vast majority are completely benign (non-cancerous) epithelial growths."
        ),
        "details": [
            "**Secretory (Functioning) Adenomas:** Produce excess hormones, causing specific endocrine conditions: Prolactinoma (menstrual irregularity, galactorrhea, fertility changes), ACTH-secreting (Cushing's disease), or Growth Hormone-secreting (Acromegaly / gigantism).",
            "**Non-Secretory (Non-Functioning) Adenomas:** Do not produce excess hormones. They are frequently discovered only when they expand upward (macroadenoma > 1 cm) and compress the optic chiasm.",
            "**Visual Field Loss (Bitemporal Hemianopsia):** Compression of the optic chiasm classically causes loss of peripheral vision in both eyes (tunnel vision).",
            "**Treatment Options:** Prolactinomas are predominantly treated with dopamine agonist medications (cabergoline or bromocriptine) without surgery. Larger macroadenomas compressing the optic apparatus are surgically removed via a minimally invasive **transsphenoidal endoscopic approach** through the nasal cavity."
        ],
        "sources": ["Endocrine Society Clinical Practice Guidelines", "NCI", "ACS"]
    },
    "notumor_context": {
        "title": "Normal MRI Scan Context & When Symptoms Persist",
        "summary": (
            "A 'No Tumor' model prediction indicates that the convolutional neural network did not detect "
            "radiological patterns characteristic of the trained space-occupying tumor classes in this specific 2D slice. "
            "However, an AI computer vision score is never an absolute ruling out of illness."
        ),
        "details": [
            "**Model Limitations:** The AI model evaluated a single 2D cross-section ($128 \\times 128$ resolution). It does not analyze volumetric 3D sequences, magnetic resonance angiography (MRA), or perfusion dynamics.",
            "**Other Neurological Conditions:** Symptoms such as chronic headaches, dizziness, or seizures can stem from migraines, tension-type headaches, vascular malformations, infections, multiple sclerosis, or metabolic causes.",
            "**Next Step:** If a patient is experiencing persistent or new neurological symptoms, an official diagnostic consultation with a board-certified neurologist or neuroradiologist remains essential, regardless of any software output."
        ],
        "sources": ["NHS UK", "American College of Radiology (ACR) Appropriateness Criteria"]
    },
    "treatments": {
        "title": "Treatment & Medical Options for Brain Tumors",
        "summary": (
            "Brain tumor management requires a multidisciplinary medical team (neurosurgeons, neuro-oncologists, "
            "radiation oncologists, neuroradiologists, and rehabilitation therapists). The optimal approach is personalized "
            "based on tumor type, WHO grade, anatomical location, molecular biomarkers, patient age, and overall health status."
        ),
        "modalities": [
            {
                "name": "1. Surgical Resection (Craniotomy & Biopsy)",
                "description": "Surgical opening of the skull to remove as much tumor tissue as safely possible (Gross Total Resection) or obtain a tissue specimen (Biopsy) for histologic and molecular testing.",
                "why_recommended": "Relieves intracranial pressure immediately, reduces tumor burden, and provides tissue for definitive molecular diagnosis.",
                "technologies": "Modern techniques include intraoperative MRI, neuro-navigation (GPS for the brain), awake craniotomy with cortical speech/motor mapping, and fluorescent dyes (5-ALA) that make tumor cells glow under blue light.",
                "side_effects": "Infection risk, bleeding, temporary neurological weakness, brain swelling (edema), seizure risk.",
                "questions_to_ask": "What are the surgical goals: complete removal or biopsy? How will eloquent brain areas (speech/movement) be protected during surgery?"
            },
            {
                "name": "2. Radiation Therapy",
                "description": "Uses high-energy X-rays, gamma rays, or charged protons to destroy cancer cell DNA, preventing tumor cells from dividing and multiplying.",
                "why_recommended": "Used after surgery to destroy remaining microscopic tumor cells, or as primary therapy when a tumor cannot be safely operated upon.",
                "types": "External Beam Radiation Therapy (EBRT), Intensity-Modulated Radiotherapy (IMRT), Stereotactic Radiosurgery (SRS / Gamma Knife / CyberKnife - delivers pinpoint radiation in 1–5 sessions), and Proton Beam Therapy.",
                "side_effects": "Fatigue, hair loss in the radiation field, skin irritation, localized brain swelling, and potential delayed cognitive changes.",
                "questions_to_ask": "What type of radiation is recommended? What are the short-term and long-term cognitive side effects, and how will they be minimized?"
            },
            {
                "name": "3. Chemotherapy",
                "description": "Cytotoxic medications administered orally or intravenously that circulate through the bloodstream to eliminate rapidly dividing cancer cells.",
                "why_recommended": "Frequently combined with radiation therapy (the Stupp protocol for Glioblastoma) to sensitize tumor cells to radiation and attack infiltrative cells.",
                "standard_agents": "Temozolomide (TMZ) is the primary oral agent because it effectively crosses the blood-brain barrier. PCV (Procarbazine, CCNU/Lomustine, Vincristine) is used for specific oligodendrogliomas.",
                "side_effects": "Nausea, fatigue, lowered blood cell counts (immunosuppression, anemia, bleeding risk), increased infection susceptibility, loss of appetite.",
                "questions_to_ask": "Is my tumor's MGMT promoter methylated (indicating higher sensitivity to temozolomide)? How often will blood counts be monitored?"
            },
            {
                "name": "4. Targeted Therapy",
                "description": "Medications that target specific molecular mutations, receptors, or blood vessel growth pathways essential for tumor proliferation, sparing normal cells.",
                "why_recommended": "Used when molecular sequencing reveals actionable genetic alterations (e.g., BRAF V600E mutations, IDH1 mutations) or to control severe peritumoral swelling.",
                "standard_agents": "Bevacizumab (Avastin) targets VEGF to stop new blood vessel formation. Vorasidenib targets mutant IDH1/2 in Grade II gliomas. BRAF inhibitors (Dabrafenib/Trametinib) for BRAF-mutated tumors.",
                "side_effects": "Elevated blood pressure, delayed wound healing, fatigue, joint pain, blood clot risks.",
                "questions_to_ask": "Did comprehensive genomic profiling reveal actionable targets? Am I eligible for targeted clinical trials?"
            },
            {
                "name": "5. Active Surveillance (Watchful Waiting)",
                "description": "Close observation through serial contrast-enhanced MRI scans (typically every 3 to 12 months) and neurological examinations without immediate invasive surgery or radiation.",
                "why_recommended": "For small, slow-growing, asymptomatic, low-grade tumors (especially incidental Grade I meningiomas or small pituitary microadenomas) where treatment risks outweigh disease risks.",
                "side_effects": "Emotional anxiety during scan intervals; no physical side effects from intervention.",
                "questions_to_ask": "What exact signs or scan changes would trigger a transition from surveillance to active treatment?"
            },
            {
                "name": "6. Supportive, Palliative & Rehabilitative Care",
                "description": "Symptom-focused medical interventions provided alongside cancer treatment to manage swelling, seizures, pain, and neurological deficits.",
                "standard_agents": "Corticosteroids (Dexamethasone) to reduce vasogenic brain edema; Anti-seizure medications (Levetiracetam/Keppra, Lamotrigine) to prevent convulsions.",
                "why_recommended": "Optimizes quality of life, protects cerebral perfusion, and maintains physical independence.",
                "side_effects": "Steroid side effects include mood changes, insomnia, elevated blood sugar, weight gain, and muscle weakness with prolonged use.",
                "questions_to_ask": "What is our plan for safely tapering steroid dosages as soon as brain swelling subsides?"
            }
        ],
        "sources": ["NCI PDQ Adult CNS Treatment", "American Cancer Society", "ESMO Clinical Practice Guidelines"]
    },
    "precautions": {
        "title": "Precautions & Patient Care Guidelines",
        "guidelines": [
            {
                "category": "Medication Compliance",
                "icon": "fa-pills",
                "text": "Take anti-seizure medications and prescribed therapies exactly as directed. Never abruptly stop anti-epileptic drugs (AEDs) or corticosteroids, as sudden cessation can trigger severe withdrawal seizures (status epilepticus) or acute adrenal insufficiency."
            },
            {
                "category": "Avoid Unproven Remedies",
                "icon": "fa-ban",
                "text": "Do not treat brain tumors with unverified herbs, megavitamin infusions, alkaline diets, or home remedies marketed as 'cures'. No diet or alternative therapy cures brain tumors, and many botanicals interact dangerously with chemotherapy or increase surgical bleeding."
            },
            {
                "category": "Seizure & Fall Safety",
                "icon": "fa-shield-halved",
                "text": "For patients experiencing seizures, dizziness, or ataxia: avoid driving or operating dangerous machinery until cleared by a physician (typically 6–12 seizure-free months). Install non-slip bathroom mats, use shower chairs, and never swim or bathe alone without supervision."
            },
            {
                "category": "Regular MRI & Oncology Follow-Ups",
                "icon": "fa-calendar-check",
                "text": "Strict adherence to surveillance neuroimaging schedules is crucial. Early detection of tumor progression or radiation necrosis allows timely, effective intervention before irreversible neurological deficits occur."
            },
            {
                "category": "Rest, Sleep & Fatigue Management",
                "icon": "fa-bed",
                "text": "Cancer-related fatigue is common after radiation and chemotherapy. Prioritize 7 to 9 hours of sleep, incorporate planned daytime rest periods, and maintain a consistent sleep schedule to support neuro-cognitive recovery."
            },
            {
                "category": "Balanced Nutrition & Hydration",
                "icon": "fa-apple-whole",
                "text": "Focus on nutrient-dense meals with adequate protein (lean meats, legumes, fish) to support cellular repair and wound healing. Stay well-hydrated. Discuss any dietary modifications with a clinical oncology dietitian."
            },
            {
                "category": "Emotional & Mental Health Support",
                "icon": "fa-heart",
                "text": "Anxiety, depression, and cognitive fatigue are recognized medical consequences of brain tumors and steroid therapy. Engage neuro-psychology services, support groups, and open family dialogue to maintain emotional well-being."
            },
            {
                "category": "Caregiver Education & Safety Net",
                "icon": "fa-users",
                "text": "Family members should learn basic seizure first-aid: gently turn the patient onto their side, clear sharp objects, never put anything in the mouth, time the seizure, and call emergency services if it lasts longer than 5 minutes."
            }
        ],
        "sources": ["NCI Supportive Care", "American Association of Neurological Surgeons (AANS)", "ACS"]
    },
    "emergencies": {
        "title": "When Should You Seek Medical Help?",
        "immediate_emergency": [
            "**First-Time Seizure / Convulsion:** Any seizure occurring in an individual without prior epilepsy history.",
            "**Prolonged Seizure (> 5 Minutes):** A seizure lasting more than five minutes, or repeated seizures without the patient regaining consciousness between episodes (Status Epilepticus).",
            "**Sudden Paralysis or Numbness:** Sudden onset of weakness, paralysis, or loss of sensation in the face, arm, or leg (especially on one side of the body).",
            "**Loss of Consciousness:** Unresponsiveness, syncope, or inability to awaken the patient.",
            "**Acute Speech Inability:** Sudden inability to speak, slurred speech, or profound confusion in understanding basic words.",
            "**Thunderclap Headache:** Sudden, explosive, unusually severe headache accompanied by stiff neck, fever, or vision blackout."
        ],
        "prompt_evaluation": [
            "New or progressively intensifying headaches, particularly headaches present upon waking or accompanied by nausea/vomiting.",
            "Gradual visual changes: persistent blurriness, double vision (diplopia), or loss of side vision (peripheral vision loss).",
            "Noticeable personality, mood, or behavioral shifts observed by family members over days or weeks.",
            "Increasing clumsiness, unsteady gait, frequent stumbling, or loss of fine motor coordination in hands.",
            "Persistent nausea, unexplained dizziness, or ringing in the ears (tinnitus)."
        ],
        "emergency_action": "In any immediate emergency, call emergency services (911 in USA/Canada, 112 in Europe, 999 in UK, or your local emergency number) immediately. Do NOT wait for an online response or delay hospital transport.",
        "sources": ["American College of Emergency Physicians (ACEP)", "NHS UK", "NCI"]
    },
    "recovery": {
        "title": "Recovery, Rehabilitation & Long-Term Support",
        "phases": [
            {
                "title": "1. Post-Operative & Acute Phase",
                "description": "Focuses on wound healing, controlling brain swelling with tapered steroids, managing pain, and monitoring for post-surgical complications or seizures in a neurological step-down unit."
            },
            {
                "title": "2. Multidisciplinary Neuro-Rehabilitation",
                "description": "Tailored therapies designed to retrain neural pathways through neuroplasticity: Physical Therapy (gait, balance, strength), Occupational Therapy (fine motor skills, daily self-care), and Speech-Language Therapy (communication, swallowing, cognition)."
            },
            {
                "title": "3. Cognitive Rehabilitation Therapy (CRT)",
                "description": "Specialized neuropsychological training to address memory deficits, concentration challenges, executive functioning, and mental processing speed affected by tumors or radiation."
            },
            {
                "title": "4. Emotional & Caregiver Wellbeing",
                "description": "Caregivers frequently experience profound physical and psychological burnout. Structured respite care, psycho-oncology counseling, and patient advocacy support groups are vital components of recovery."
            },
            {
                "title": "5. Returning to Daily Life, School & Employment",
                "description": "Transitioning back to work should be gradual (part-time or modified duties). Neuro-oncology social workers assist with disability accommodations, vocational rehabilitation, and legal workplace protections."
            }
        ],
        "sources": ["World Federation for NeuroRehabilitation (WFNR)", "NCI", "ACS"]
    },
    "questions_for_doctor": {
        "title": "Essential Questions to Ask Your Medical Team",
        "questions": [
            "What exact type and WHO grade is my brain tumor, and where is it located?",
            "What molecular or genetic biomarker testing (IDH, 1p/19q, MGMT) was performed on my biopsy?",
            "What are the goals of my proposed treatment: complete removal, long-term disease control, or symptom relief?",
            "What are the benefits, potential risks, and permanent side effects of each treatment option?",
            "How might this surgery, radiation, or chemotherapy affect my memory, speech, vision, or motor skills?",
            "Will I need anti-seizure medication or steroids, and how will their dosage be adjusted over time?",
            "What specific warning symptoms or red flags mean I should go to the emergency room immediately?",
            "Are there clinical trials (new targeted therapies, immunotherapy, tumor treating fields) available for my tumor profile?",
            "What rehabilitation services (physical, speech, occupational therapy) should I start immediately?",
            "Can you refer me for a second opinion with a specialized comprehensive neuro-oncology center?"
        ],
        "sources": ["American Society of Clinical Oncology (ASCO)", "NCI", "ACS"]
    }
}


def build_system_prompt():
    """Builds clinical safety and pedagogical prompt for live LLM generation."""
    return (
        "You are the NeuroScan AI Brain Tumor Care Assistant, an empathetic, highly knowledgeable, "
        "and medically responsible educational health specialist. Your mission is to provide clear, "
        "accurate, and comforting health information to brain tumor patients, families, and caregivers.\n\n"
        "STRICT CLINICAL RULES:\n"
        "1. Never provide a clinical diagnosis or suggest that an AI computer vision prediction is a confirmed medical fact.\n"
        "2. Never prescribe medications, suggest drug dosages, or endorse unproven alternative therapies/cures.\n"
        "3. Emphasize that brain tumors vary widely by type, WHO grade, molecular biomarkers, and patient factors.\n"
        "4. In any question mentioning acute emergency red flags (first seizure, prolonged seizure, sudden paralysis, "
        "loss of consciousness), immediately instruct the user to call emergency medical services (911/112/local EMS).\n"
        "5. Structure your response with clean markdown headings, bullet points, and simple language.\n"
        "6. Cite authoritative sources (NCI, WHO, ACS, NHS) and advise discussing with a qualified neurologist, neurosurgeon, or neuro-oncologist.\n"
    )


def contains_word(words, text):
    """Accurate word-boundary regex match to prevent substring clashes (e.g. 'eat' in 'treatment')."""
    pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def synthesize_nlp_response(user_message, context_category=None):
    """
    Intelligent medical NLP response synthesizer.
    Matches queries against the verified medical knowledge base (NCI/WHO/ACS)
    and formats empathetic, structured answers with headings and bullet points.
    """
    msg = user_message.lower().strip()

    # 1. EMERGENCY CHECK (Highest Priority)
    emergency_keywords = ["emergency", "seizure", "seizures", "convulsion", "convulsions", "unconscious",
                          "passed out", "sudden paralysis", "can't speak", "cant speak", "paralyzed",
                          "thunderclap", "collapsed", "fainted"]
    if contains_word(emergency_keywords, msg) or "passed out" in msg or "sudden paralysis" in msg:
        return {
            "reply": (
                "### 🚨 Immediate Emergency Guidance\n\n"
                "**If you or someone around you is experiencing a medical emergency, call your local emergency services (911 in US/Canada, 112 in Europe, 999 in UK) or go to the nearest emergency room immediately.**\n\n"
                "#### Immediate Emergency Warning Signs (Do Not Wait):\n"
                "- **First-Time Seizure:** Any new seizure in an individual with no previous history.\n"
                "- **Prolonged Seizure (> 5 Minutes):** Or repeated seizures without full recovery between them.\n"
                "- **Sudden Paralysis or Numbness:** Rapid loss of motor function or sensation on one side of the body.\n"
                "- **Loss of Consciousness:** Inability to awaken or sudden unresponsiveness.\n"
                "- **Sudden Inability to Speak:** Inability to articulate or comprehend speech.\n"
                "- **Severe Sudden Headache:** An explosive, catastrophic headache with stiff neck or vomiting.\n\n"
                "> [!IMPORTANT]\n"
                "> Never delay emergency medical care to wait for an AI response or online information."
            ),
            "sources": ["American College of Emergency Physicians", "NHS UK", "NCI"],
            "suggested_followups": [
                "What is basic seizure first aid?",
                "What are routine warning signs vs emergencies?",
                "What questions should I ask my doctor about seizures?"
            ]
        }

    # 2. DIET, NUTRITION, SUPPLEMENTS & CURE MYTH CHECK
    diet_keywords = ["diet", "diets", "food", "foods", "nutrition", "eat", "eating", "sugar", "keto",
                     "ketogenic", "supplement", "supplements", "vitamin", "vitamins", "remedy", "remedies",
                     "cure", "cures", "herb", "herbal", "alkaline", "fasting"]
    if contains_word(diet_keywords, msg) or "home remedy" in msg or "alternative cure" in msg:
        return {
            "reply": (
                "### 🥗 Diet, Nutrition & Lifestyle Guidance for Brain Tumor Patients\n\n"
                "Good nutrition supports healing, helps maintain muscle mass, reduces treatment fatigue, and enhances immune defense.\n\n"
                "#### Evidence-Based Nutritional Advice:\n"
                "- **Adequate Protein Intake:** Incorporate lean poultry, fish, eggs, tofu, beans, and lentils to facilitate cellular recovery and tissue repair after surgery or radiation.\n"
                "- **Hydration:** Aim for 8 to 10 glasses of water daily. Proper hydration minimizes headaches, counteracts medication dry mouth, and supports kidney filtration.\n"
                "- **Managing Treatment Nausea:** Eat small, frequent meals rather than large meals. Ginger tea, dry crackers, and cold foods are often better tolerated.\n"
                "- **Caloric Maintenance:** Involuntary weight loss is common during chemoradiation. Focus on nutrient-dense foods (avocados, nut butters, olive oil, smoothies).\n\n"
                "#### Important Truth Regarding 'Cures' & Alternative Remedies:\n"
                "- **No Diet or Remedy Cures Brain Tumors:** **No food, vitamin, supplement, ketogenic diet, alkaline water, herbal extract, yoga practice, or home remedy cures brain tumors.** Brain tumors require specialized clinical medical therapies (surgery, radiotherapy, targeted therapy).\n"
                "- **Myth: 'Sugar feeds cancer'** — While limiting refined sugars is good for overall metabolic health, drastically starving the body of carbohydrates can lead to dangerous malnutrition.\n"
                "- **Supplement Dangers:** High doses of antioxidants (vitamins C, E) or herbs like St. John's Wort can neutralize the intended cell-killing mechanism of radiation therapy or interfere with anti-epileptic drug metabolism.\n\n"
                "> [!WARNING]\n"
                "> Never delay or replace standard oncology treatment with unproven alternative diets or remedies. Always consult your oncology team or a registered clinical oncology dietitian before taking any supplement."
            ),
            "sources": ["Oncology Nutrition Dietetic Practice Group (AND)", "American Cancer Society Nutrition Guidelines", "NCI"],
            "suggested_followups": [
                "What precautions should be taken with supplements?",
                "How to manage fatigue after radiation therapy?",
                "What treatments are available for brain tumors?"
            ]
        }

    # 3. PRECAUTIONS & CARE
    precaution_keywords = ["precaution", "precautions", "safety", "prevention", "protect", "lifestyle", "routine", "caregiver", "family"]
    if contains_word(precaution_keywords, msg):
        guidelines = KNOWLEDGE_BASE["precautions"]["guidelines"]
        text = "### 🛡️ Essential Precautions & Patient Care Guidelines\n\n"
        text += "Caring for someone diagnosed with a brain tumor requires a structured, safety-first routine. Based on clinical guidance from the **National Cancer Institute (NCI)** and **American Association of Neurological Surgeons (AANS)**:\n\n"
        for g in guidelines:
            text += f"- **{g['category']}:** {g['text']}\n"
        text += "\n> [!WARNING]\n"
        text += "> **No food, supplement, herbal extract, or home remedy cures brain tumors.** Always consult your neuro-oncologist before changing diet, exercise, or medications.\n"
        return {
            "reply": text,
            "sources": ["National Cancer Institute (NCI)", "AANS Patient Safety Guidelines", "ACS"],
            "suggested_followups": [
                "What foods should a brain tumor patient eat?",
                "What treatments are available for brain tumors?",
                "How can family members support a patient?"
            ]
        }

    # 4. TREATMENT OPTIONS
    treatment_keywords = ["treatment", "treatments", "treat", "treated", "surgery", "radiation", "radiosurgery",
                          "chemotherapy", "therapy", "therapies", "medicine", "medicines", "operation", "craniotomy",
                          "biopsy", "temozolomide", "avastin", "gamma knife"]
    if contains_word(treatment_keywords, msg):
        treatments = KNOWLEDGE_BASE["treatments"]["modalities"]
        text = "### 🩺 Available Treatment & Medical Options\n\n"
        text += "Brain tumor treatment is personalized and determined by a multidisciplinary tumor board based on tumor type, WHO grade, molecular biomarkers, and anatomical location:\n\n"
        for t in treatments:
            text += f"#### {t['name']}\n"
            text += f"- **What it means:** {t['description']}\n"
            text += f"- **Why recommended:** {t['why_recommended']}\n"
            text += f"- **Possible side effects:** {t['side_effects']}\n"
            text += f"- **Key question to ask:** *\"{t['questions_to_ask']}\"*\n\n"
        text += "> [!NOTE]\n"
        text += "> Treatment recommendations must come directly from your neurosurgical and oncology team. Machine learning model predictions cannot determine treatment pathways.\n"
        return {
            "reply": text,
            "sources": ["National Cancer Institute (NCI PDQ)", "American Cancer Society", "ESMO Guidelines"],
            "suggested_followups": [
                "What is the difference between craniotomy and biopsy?",
                "What are targeted therapies and immunotherapy for brain tumors?",
                "What questions should I ask my doctor about surgery?"
            ]
        }

    # 4. GLIOMA SPECIFICS
    if "glioma" in msg or context_category == "glioma":
        g = KNOWLEDGE_BASE["glioma"]
        text = f"### 🧠 {g['title']}\n\n"
        text += f"{g['summary']}\n\n"
        text += "#### Key Clinical Aspects:\n"
        for d in g["details"]:
            text += f"- {d}\n"
        text += "\n#### Standard Treatment Framework:\n"
        text += "- **Maximal Safe Surgical Resection:** Removing as much tumor as possible while preserving neurological function.\n"
        text += "- **Stupp Protocol (High-Grade):** Concurrent Temozolomide (oral chemotherapy) with radiation therapy, followed by adjuvant Temozolomide.\n"
        text += "- **Targeted & Molecular Therapy:** IDH inhibitors (e.g., Vorasidenib) for IDH-mutant tumors, Tumor Treating Fields (TTFields), and clinical trial enrollment.\n"
        return {
            "reply": text,
            "sources": ["WHO 2021 Classification", "NCI Adult Glioma Treatment Guidelines", "ACS"],
            "suggested_followups": [
                "What are the warning signs of glioma progression?",
                "What precautions should a glioma patient take?",
                "Can a patient recover after glioma surgery?"
            ]
        }

    # 5. MENINGIOMA SPECIFICS
    if "meningioma" in msg or context_category == "meningioma":
        m = KNOWLEDGE_BASE["meningioma"]
        text = f"### 🛡️ {m['title']}\n\n"
        text += f"{m['summary']}\n\n"
        text += "#### Key Clinical Aspects:\n"
        for d in m["details"]:
            text += f"- {d}\n"
        text += "\n#### Primary Management Options:\n"
        text += "- **Active Surveillance (Watchful Waiting):** Periodic contrast MRI scans (every 6–12 months) for small, asymptomatic Grade I meningiomas.\n"
        text += "- **Surgical Resection (Craniotomy):** Recommended when tumors are symptomatic, large, or exerting significant mass effect.\n"
        text += "- **Stereotactic Radiosurgery (SRS / Gamma Knife):** Highly effective pinpoint radiation for surgically inaccessible, recurrent, or residual tumors.\n"
        return {
            "reply": text,
            "sources": ["National Cancer Institute", "American Cancer Society", "NHS UK"],
            "suggested_followups": [
                "What symptoms indicate a meningioma is growing?",
                "What is stereotactic radiosurgery for meningioma?",
                "What precautions should be taken after meningioma surgery?"
            ]
        }

    # 6. PITUITARY SPECIFICS
    if "pituitary" in msg or context_category == "pituitary":
        p = KNOWLEDGE_BASE["pituitary"]
        text = f"### 🔬 {p['title']}\n\n"
        text += f"{p['summary']}\n\n"
        text += "#### Key Clinical Aspects:\n"
        for d in p["details"]:
            text += f"- {d}\n"
        text += "\n#### Common Management Pathways:\n"
        text += "- **Medical Therapy:** Dopamine agonists (Cabergoline, Bromocriptine) are first-line for Prolactinomas and frequently shrink the tumor without surgery.\n"
        text += "- **Transsphenoidal Endoscopic Surgery:** Minimally invasive removal through the nasal passages for non-functioning or vision-compressing adenomas.\n"
        text += "- **Hormone Replacement & Monitoring:** Long-term endocrinology follow-ups to maintain optimal thyroid, cortisol, and reproductive hormone levels.\n"
        return {
            "reply": text,
            "sources": ["Endocrine Society Guidelines", "NCI Pituitary Tumors PDQ", "ACS"],
            "suggested_followups": [
                "How does a pituitary tumor affect vision?",
                "What is transsphenoidal surgery?",
                "What questions should I ask an endocrinologist?"
            ]
        }

    # 7. DIET, NUTRITION & LIFESTYLE
    if any(k in msg for k in ["diet", "food", "nutrition", "eat", "sugar", "keto", "supplement", "vitamin", "exercise", "sleep"]):
        return {
            "reply": (
                "### 🥗 Diet, Nutrition & Lifestyle Guidance for Brain Tumor Patients\n\n"
                "Good nutrition supports healing, helps maintain muscle mass, reduces treatment fatigue, and enhances immune defense.\n\n"
                "#### Evidence-Based Nutritional Advice:\n"
                "- **Adequate Protein Intake:** Incorporate lean poultry, fish, eggs, tofu, beans, and lentils to facilitate cellular recovery and tissue repair after surgery or radiation.\n"
                "- **Hydration:** Aim for 8 to 10 glasses of water daily. Proper hydration minimizes headaches, counteracts medication dry mouth, and supports kidney filtration.\n"
                "- **Managing Treatment Nausea:** Eat small, frequent meals rather than large meals. Ginger tea, dry crackers, and cold foods are often better tolerated.\n"
                "- **Caloric Maintenance:** Involuntary weight loss is common during chemoradiation. Focus on nutrient-dense foods (avocados, nut butters, olive oil, smoothies).\n\n"
                "#### Important Dietary Cautions:\n"
                "- **Myth: 'Sugar feeds cancer'** — While limiting refined sugars is good for overall metabolic health, drastically starving the body of carbohydrates can lead to dangerous malnutrition.\n"
                "- **Myth: 'Keto or alkaline diets cure tumors'** — Clinical trials evaluating ketogenic diets are experimental and ongoing; never attempt extreme diets without your neuro-oncologist's direct supervision.\n"
                "- **High-Dose Antioxidant Supplements:** High doses of vitamins C, E, or St. John's Wort can neutralize the oxidative mechanism of radiation therapy or interfere with anti-epileptic drugs.\n\n"
                "> [!TIP]\n"
                "> Request a referral to a certified **Board-Certified Specialist in Oncology Nutrition (CSO)** through your cancer center."
            ),
            "sources": ["Oncology Nutrition Dietetic Practice Group (AND)", "American Cancer Society Nutrition Guidelines", "NCI"],
            "suggested_followups": [
                "What precautions should be taken with supplements?",
                "How to manage fatigue after radiation therapy?",
                "What general precautions should a patient take?"
            ]
        }

    # 8. QUESTIONS FOR THE DOCTOR
    if any(k in msg for k in ["question", "ask doctor", "ask neurologist", "consult", "oncologist", "appointment"]):
        questions = KNOWLEDGE_BASE["questions_for_doctor"]["questions"]
        text = "### 📋 Essential Questions to Ask Your Neurologist or Oncologist\n\n"
        text += "Taking a trusted family member or advocate and an organized list of questions to appointments helps ensure all critical aspects are addressed:\n\n"
        for idx, q in enumerate(questions, 1):
            text += f"{idx}. **{q}**\n"
        text += "\n> [!TIP]\n"
        text += "> Ask your physician if you can record the audio of your consultation on your phone so you can review complex medical explanations calmly with your family later."
        return {
            "reply": text,
            "sources": ["American Society of Clinical Oncology (ASCO)", "NCI", "ACS"],
            "suggested_followups": [
                "What is a second opinion and when should I get one?",
                "What treatments are available for brain tumors?",
                "What are the warning signs of complications?"
            ]
        }

    # 9. RECOVERY & REHABILITATION
    if any(k in msg for k in ["recover", "rehabilitation", "rehab", "after surgery", "therapy", "work", "school", "long-term"]):
        phases = KNOWLEDGE_BASE["recovery"]["phases"]
        text = "### 🏃 Recovery, Rehabilitation & Returning to Daily Life\n\n"
        text += "Recovery from brain tumor treatment is unique to every individual. Brain tissue heals gradually through **neuroplasticity** (the brain's ability to reorganize and form new neural connections):\n\n"
        for p in phases:
            text += f"#### {p['title']}\n"
            text += f"{p['description']}\n\n"
        text += "> [!NOTE]\n"
        text += "> Progress may be non-linear, with good days and challenging days. Celebrate small functional milestones and engage neuro-psychology services whenever emotional fatigue arises."
        return {
            "reply": text,
            "sources": ["World Federation for NeuroRehabilitation", "NCI Supportive Care", "ACS"],
            "suggested_followups": [
                "What questions should I ask my physical therapist?",
                "What precautions should I take during post-surgery recovery?",
                "How can caregivers prevent burnout?"
            ]
        }

    # 10. BENIGN VS MALIGNANT
    if any(k in msg for k in ["benign", "malignant", "cancer", "non-cancerous", "difference"]):
        return {
            "reply": (
                "### ⚖️ Benign vs. Malignant Brain Tumors: Clinical Distinctions\n\n"
                "In general oncology, 'benign' means non-cancerous and 'malignant' means cancerous. Inside the brain, however, both require careful medical evaluation due to the confined space of the skull.\n\n"
                "| Feature | Benign Brain Tumors (e.g., Grade I Meningioma) | Malignant Brain Tumors (e.g., Glioblastoma) |\n"
                "| :--- | :--- | :--- |\n"
                "| **Growth Rate** | Slow and progressive over months or years | Rapid and aggressive over weeks or months |\n"
                "| **Boundaries** | Distinct, encapsulated (extra-axial) | Diffuse, infiltrative into surrounding brain parenchyma |\n"
                "| **Surgical Resection** | Often completely curable if fully resectable | Microscopic infiltration makes complete removal challenging |\n"
                "| **Recurrence Risk** | Low following complete gross total resection | High; typically requires adjuvant radiation and chemotherapy |\n"
                "| **Clinical Impact** | Causes symptoms via mechanical compression | Causes symptoms via compression, infiltration, and tissue destruction |\n\n"
                "> [!IMPORTANT]\n"
                "> Even benign tumors in the brain can produce serious neurological symptoms if located near critical cranial nerves, the brainstem, or major blood vessels."
            ),
            "sources": ["World Health Organization (WHO)", "National Cancer Institute (NCI)", "American Cancer Society"],
            "suggested_followups": [
                "What is active surveillance for benign tumors?",
                "What treatments are available for malignant brain tumors?",
                "What are common symptoms of brain tumors?"
            ]
        }

    # 11. GENERAL TUMOR EXPLANATION / DEFAULT EMPATHETIC FALLBACK
    return {
        "reply": (
            "### 🧠 Brain Tumor Health & Educational Guidance\n\n"
            "Thank you for reaching out. A brain tumor diagnosis or exploration can feel overwhelming. "
            "Here are the foundational facts established by the **National Cancer Institute (NCI)** and **World Health Organization (WHO)**:\n\n"
            "- **What it is:** An abnormal growth of cells in the brain or surrounding meningeal membranes. Tumors can be primary (originating in the brain) or metastatic (originating elsewhere).\n"
            "- **Diagnosis Requires Clinical Confirmation:** Definitive diagnosis always requires a multi-parametric contrast MRI and, in most cases, a surgical biopsy for histological and molecular sequencing.\n"
            "- **Personalized Treatment:** Therapies (surgery, radiosurgery, chemotherapy, targeted agents) are tailored specifically to tumor grade, molecular markers (IDH, 1p/19q, MGMT), location, and patient health.\n"
            "- **Holistic Care:** Medication adherence (anti-seizure drugs, steroids), fall/seizure precautions, balanced nutrition, and cognitive rehabilitation are essential pillars of wellbeing.\n\n"
            "> [!IMPORTANT]\n"
            "> **Medical Disclaimer:** This system provides educational health information and cannot diagnose or prescribe treatment. "
            "Always consult your neurologist, neurosurgeon, or neuro-oncologist regarding specific symptoms and clinical care decisions."
        ),
        "sources": ["National Cancer Institute (NCI)", "World Health Organization (WHO)", "American Cancer Society"],
        "suggested_followups": [
            "What treatments are available for brain tumors?",
            "What precautions should a brain tumor patient take?",
            "What are emergency warning signs?",
            "What questions should I ask my doctor?"
        ]
    }


def query_gemini_api(user_message, context_category=None, api_key=None):
    """
    Optional live generation hook via Gemini REST API when GEMINI_API_KEY is configured.
    Falls back gracefully to synthesize_nlp_response() on any failure.
    """
    if not api_key:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        system_prompt = build_system_prompt()
        if context_category:
            system_prompt += f"\nOptional Context: The user's MRI test scan was classified under the category: '{context_category}'. Remember this is an educational AI demonstration, not a medical diagnosis."

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"System Context:\n{system_prompt}\n\nUser Question:\n{user_message}"}]}
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800
            }
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text_reply = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text_reply:
                    return {
                        "reply": text_reply,
                        "sources": ["National Cancer Institute (NCI)", "World Health Organization (WHO)", "American Cancer Society"],
                        "suggested_followups": [
                            "What treatments are available for brain tumors?",
                            "What precautions should a brain tumor patient take?",
                            "What questions should I ask my neurologist?"
                        ]
                    }
    except Exception as e:
        logger.warning(f"Live Gemini API query failed, using built-in verified knowledge engine: {e}")

    return None


def get_ai_care_response(user_message, context_category=None):
    """
    Primary API entry point for the AI Care Assistant.
    1. Checks for optional GEMINI_API_KEY.
    2. Uses live Gemini if available and valid.
    3. Seamlessly defaults to verified medical knowledge base (NCI/WHO/ACS).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        live_res = query_gemini_api(user_message, context_category=context_category, api_key=api_key)
        if live_res:
            return live_res

    return synthesize_nlp_response(user_message, context_category=context_category)


def get_curated_care_data():
    """Returns structured educational datasets for initial care page rendering."""
    return {
        "knowledge_base": KNOWLEDGE_BASE,
        "sources": MEDICAL_SOURCES,
        "precautions": KNOWLEDGE_BASE["precautions"]["guidelines"],
        "treatments": KNOWLEDGE_BASE["treatments"]["modalities"],
        "emergencies": KNOWLEDGE_BASE["emergencies"],
        "recovery": KNOWLEDGE_BASE["recovery"]["phases"],
        "questions": KNOWLEDGE_BASE["questions_for_doctor"]["questions"]
    }
