# save as 1_generate_synthetic_data.py - Enhanced with 70+ skills and 50+ certifications

import pandas as pd
import numpy as np
import random

company_tiers = {
    'google': 1, 'microsoft': 1, 'amazon': 1, 'apple': 1, 'meta': 1,
    'netflix': 1, 'linkedin': 1, 'adobe': 1, 'salesforce': 1, 'uber': 1,
    'ibm': 2, 'accenture': 2, 'oracle': 2, 'sap': 2, 'deloitte': 2,
    'tcs': 3, 'infosys': 3, 'wipro': 3, 'cts': 3, 'hcl': 3,
    'virtusa': 4, 'hexaware': 4, 'zensar': 4,
    'coforge': 5, 'valuelabs': 5, 'happiest-minds': 5,
}

# Enhanced roles - 35+ different IT positions
roles = [
    'Software Developer', 'Senior Developer', 'Tech Lead', 'QA Engineer',
    'Data Scientist', 'DevOps Engineer', 'Cloud Engineer', 'Business Analyst',
    'Machine Learning Engineer', 'Product Manager', 'Security Engineer',
    'AI Researcher', 'Solution Architect', 'Full Stack Developer',
    'Database Administrator', 'UX/UI Designer', 'Project Manager', 'Data Analyst',
    'Business Intelligence Analyst', 'Network Engineer', 'System Administrator',
    'Mobile App Developer', 'Blockchain Developer', 'Site Reliability Engineer',
    'Cybersecurity Specialist', 'Cloud Architect', 'Scrum Master', 'Technical Lead',
    'Performance Engineer', 'Infrastructure Engineer', 'Software Architect',
    'Data Engineer', 'Platform Engineer', 'Release Manager', 'Quality Assurance Lead'
]

locations = ['Bangalore', 'Hyderabad', 'Chennai', 'Mumbai', 'Pune', 'Delhi', 'Gurgaon', 'Noida', 'Kolkata']

education_levels = ['Bachelor', 'Master', 'PhD']

# 50+ Certifications for 2025
certifications = [
    'None',
    'AWS Certified Solutions Architect - Associate',
    'AWS Certified Solutions Architect - Professional', 
    'AWS Certified Developer - Associate',
    'AWS Certified SysOps Administrator - Associate',
    'AWS Certified DevOps Engineer - Professional',
    'AWS Certified Security - Specialty',
    'AWS Certified Machine Learning - Specialty',
    'AWS Certified Data Analytics - Specialty',
    'AWS Cloud Practitioner',
    'Microsoft Certified: Azure Fundamentals',
    'Microsoft Certified: Azure Administrator Associate',
    'Microsoft Certified: Azure Solutions Architect Expert',
    'Microsoft Certified: Azure DevOps Engineer Expert',
    'Microsoft Certified: Azure Security Engineer Associate',
    'Microsoft Certified: Azure AI Engineer Associate',
    'Microsoft Certified: Azure Data Scientist Associate',
    'Microsoft Certified: Azure Data Engineer Associate',
    'Google Cloud Professional Cloud Architect',
    'Google Cloud Professional Data Engineer',
    'Google Cloud Professional Machine Learning Engineer',
    'Google Cloud Professional Security Engineer',
    'Google Cloud Associate Cloud Engineer',
    'Google Cloud Professional DevOps Engineer',
    'Certified Information Systems Security Professional (CISSP)',
    'Certified Information Security Manager (CISM)',
    'Certified Information Systems Auditor (CISA)',
    'Certified Ethical Hacker (CEH)',
    'CompTIA Security+',
    'CompTIA Network+',
    'CompTIA A+',
    'CompTIA Cloud+',
    'CompTIA CySA+',
    'Cisco Certified Network Associate (CCNA)',
    'Cisco Certified Network Professional (CCNP)',
    'Cisco Certified Internetwork Expert (CCIE)',
    'Project Management Professional (PMP)',
    'Certified Scrum Master (CSM)',
    'Professional Scrum Master (PSM)',
    'SAFe Agilist Certification',
    'ITIL Foundation',
    'ITIL Expert',
    'Docker Certified Associate (DCA)',
    'Certified Kubernetes Administrator (CKA)',
    'Certified Kubernetes Application Developer (CKAD)',
    'Certified Kubernetes Security Specialist (CKS)',
    'TensorFlow Developer Certificate',
    'Oracle Database Administrator Certified Professional',
    'Oracle MySQL Database Administration',
    'Salesforce Certified Administrator',
    'Salesforce Certified Platform Developer',
    'Certified Cloud Security Professional (CCSP)',
    'Certificate of Cloud Security Knowledge (CCSK)',
    'VMware Certified Professional (VCP)',
    'Red Hat Certified Engineer (RHCE)',
    'Certified Data Professional (CDP)'
]

# 70+ Skills for 2025
skills = [
    'Cloud Computing', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Data Science',
    'Cybersecurity', 'DevOps', 'Full Stack Development', 'Data Engineering', 'Big Data',
    'Natural Language Processing', 'Computer Vision', 'MLOps', 'Cloud Security', 'AWS',
    'Microsoft Azure', 'Google Cloud Platform', 'Kubernetes', 'Docker', 'Terraform',
    'Ansible', 'Jenkins', 'Python', 'Java', 'JavaScript', 'TypeScript', 'React',
    'Angular', 'Vue.js', 'Node.js', 'Spring Boot', 'Django', 'Flask', 'FastAPI',
    'Microservices', 'REST APIs', 'GraphQL', 'Database Management', 'SQL', 'NoSQL',
    'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'ElasticSearch', 'Apache Kafka',
    'Apache Spark', 'Hadoop', 'Tableau', 'Power BI', 'Data Visualization', 'Blockchain',
    'Cryptocurrency', 'Smart Contracts', 'IoT', 'Edge Computing', 'Serverless Architecture',
    'Lambda Functions', 'API Gateway', 'Message Queues', 'Event-Driven Architecture',
    'CI/CD', 'Git', 'GitHub Actions', 'GitLab CI', 'Agile', 'Scrum', 'Kanban',
    'Project Management', 'Technical Writing', 'Code Review', 'Test Automation',
    'Unit Testing', 'Integration Testing', 'Performance Testing', 'Load Testing',
    'Security Testing', 'Penetration Testing', 'Vulnerability Assessment', 'Network Security',
    'Information Security', 'Identity Management', 'OAuth', 'SAML', 'Single Sign-On'
]

data_rows = []

for _ in range(3000):  # generate 3000 samples
    current_company = random.choice(list(company_tiers.keys()))
    target_company = random.choice([c for c in company_tiers if c != current_company])
    current_tier = company_tiers[current_company]
    target_tier = company_tiers[target_company]
    current_role = random.choice(roles)
    gender = random.choices(['Male', 'Female', 'Other'], weights=[0.65, 0.33, 0.02])[0]
    location = random.choice(locations)
    education = random.choice(education_levels)
    cert = random.choice(certifications)
    skill = random.choice(skills)
    years_exp = round(random.uniform(1, 15), 1)
    base_salary = random.randint(400000, 3000000)
    current_ctc = int(base_salary + years_exp * 40000 + random.randint(-50000, 50000))
    hike_factor = random.uniform(1.2, 1.6)
    expected_salary = int(current_ctc * hike_factor)

    data_rows.append({
        'current_company': current_company,
        'target_company': target_company,
        'years_of_experience': years_exp,
        'current_salary': current_ctc,
        'expected_salary': expected_salary,
        'gender': gender,
        'location': location,
        'current_role': current_role,
        'sector': 'IT',
        'education': education,
        'certifications': cert,
        'skills': skill,
        'current_company_tier': current_tier,
        'target_company_tier': target_tier
    })

df = pd.DataFrame(data_rows)
df.to_csv('synthetic_salary_data.csv', index=False)
print("✅ Enhanced synthetic dataset saved as 'synthetic_salary_data.csv'")
print(f"📊 Generated {len(data_rows)} records with {len(skills)} skills and {len(certifications)} certifications")
print(f"🎯 Skills Count: {len(skills)}")
print(f"🎓 Certifications Count: {len(certifications)}")
print(f"💼 Roles Count: {len(roles)}")