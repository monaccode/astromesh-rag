"""Generate the Nimbus corpus: the documentation of a fictional cloud-storage product.

Stdlib only and seeded, so `python generate.py [out.jsonl]` always writes the same bytes.
Every document carries a `trap` tag naming the retrieval difficulty it was written to create:

- exact       error articles and plan SKUs; sibling codes share almost all of their wording
- paraphrase  billing/account FAQs in formal vocabulary that colloquial questions do not use
- distractor  policies in two versions (2024, 2026) that differ only in the deciding facts
- noise       blog posts and release notes that reuse product words but answer nothing
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SEED = 20261014
rng = random.Random(SEED)
DOCS: list[dict] = []


def add(doc_id, text, type_, trap, lang, version=None):
    meta = {"type": type_, "trap": trap, "lang": lang}
    if version:
        meta["version"] = version
    DOCS.append({"id": doc_id, "text": text.strip(), "metadata": meta})


# --------------------------------------------------------------------------- error articles
# Each entry: codes, lang, title/cause/fix templates with {0} {1} {2} slots, one slot tuple per
# code. Siblings share the template and the headings, so only the slots tell them apart.

ERROR_HEADINGS = {
    "es": [
        ("Qué significa", "Cómo resolverlo"),
        ("Causa", "Solución"),
        ("Por qué aparece", "Qué hacer"),
    ],
    "en": [
        ("What it means", "How to fix it"),
        ("Cause", "Resolution"),
        ("Why you see it", "What to do"),
    ],
}

ERRORS = [
    (
        ("4012", "4013"),
        "es",
        "La sesión {0}",
        "El cliente intentó sincronizar con un token de sesión que {1}. El servidor rechaza la "
        "petición y la sincronización queda en pausa hasta que la cuenta vuelva a autenticarse.",
        "Cierra sesión en la aplicación y entra de nuevo con tu correo y tu contraseña. {2} Los "
        "cambios pendientes se suben automáticamente en cuanto la sesión se recupera.",
        [
            (
                "expiró",
                "superó los 30 días sin actividad en ese dispositivo",
                "No hace falta reinstalar la aplicación ni volver a vincular la carpeta.",
            ),
            (
                "fue revocada",
                "un administrador revocó desde la consola del equipo",
                "Si el administrador retiró el acceso a propósito, tendrá que restituirlo antes de que "
                "puedas volver a entrar.",
            ),
        ],
    ),
    (
        ("4031", "4032"),
        "es",
        "Cuota de almacenamiento {0} agotada",
        "La subida no se completó porque {1} llegó al 100 %. Los archivos que ya estaban guardados "
        "siguen disponibles para leer y descargar, pero no se aceptan archivos nuevos ni nuevas "
        "versiones.",
        "{2} La sincronización se reanuda sola cuando vuelve a haber espacio.",
        [
            (
                "personal",
                "el espacio de tu cuenta personal",
                "Libera espacio vaciando la papelera, que también cuenta para la cuota, o amplía la "
                "capacidad desde Configuración > Plan.",
            ),
            (
                "del equipo",
                "el espacio compartido del equipo",
                "Solo un administrador puede ampliar la capacidad del equipo desde la consola; mientras "
                "tanto, mover archivos a tu espacio personal no libera la cuota compartida.",
            ),
        ],
    ),
    (
        ("4040", "4041"),
        "es",
        "Archivo no encontrado ({0})",
        "La aplicación pidió un archivo que {1}. Suele pasar con accesos directos o enlaces "
        "guardados antes del cambio.",
        "Comprueba primero que tienes abierta la cuenta correcta. {2}",
        [
            (
                "en la papelera",
                "se movió a la papelera desde otro dispositivo o desde la web",
                "Ábrelo en la papelera de la web y elige Restaurar; vuelve a su carpeta original y "
                "conserva sus enlaces.",
            ),
            (
                "eliminado",
                "se eliminó definitivamente y ya no está en la papelera",
                "No se puede restaurar desde la papelera. Si el plan incluye retención avanzada, un "
                "administrador puede recuperarlo desde Registros > Eliminados.",
            ),
        ],
    ),
    (
        ("4290", "4291"),
        "es",
        "Demasiadas peticiones ({0})",
        "La integración superó {1} asignado a la clave de API. Las peticiones siguientes se "
        "rechazan con HTTP 429 hasta que la ventana se renueva.",
        "{2} Agrupar operaciones con el endpoint de lotes reduce el número de llamadas.",
        [
            (
                "por minuto",
                "el límite de peticiones por minuto",
                "Espera unos segundos y reintenta; la cabecera Retry-After indica cuántos.",
            ),
            (
                "por día",
                "el límite diario de peticiones",
                "El contador se reinicia a las 00:00 UTC. Si el límite se queda corto de forma "
                "habitual, solicita una ampliación desde la consola de desarrolladores.",
            ),
        ],
    ),
    (
        ("5003", "5004"),
        "es",
        "Fallo al subir un fragmento ({0})",
        "Los archivos grandes se suben en fragmentos de 8 MB. Este error indica que un fragmento "
        "{1}.",
        "La aplicación reintenta el fragmento tres veces antes de mostrar el aviso. {2}",
        [
            (
                "suma de verificación",
                "no coincide con la suma de verificación que calculó el cliente, es decir, los datos se "
                "alteraron en tránsito",
                "Suele deberse a un proxy o antivirus que inspecciona el tráfico; excluye el dominio de "
                "Nimbus de la inspección y vuelve a intentarlo.",
            ),
            (
                "tiempo agotado",
                "llegó después de los 120 segundos que el servidor espera por cada fragmento",
                "Suele deberse a una conexión lenta o inestable; reduce el tamaño del fragmento en "
                "Preferencias > Red y vuelve a intentarlo.",
            ),
        ],
    ),
    (
        ("2101", "2102"),
        "es",
        "Conflicto de sincronización: {0}",
        "Nimbus detectó que {1} antes de que cualquiera de las dos versiones terminara de "
        "sincronizar.",
        "No se pierde nada: {2}. Revisa ambas, quédate con la correcta y elimina la otra.",
        [
            (
                "ediciones simultáneas",
                "dos dispositivos guardaron cambios distintos sobre el mismo archivo",
                "la segunda versión se guarda como copia en conflicto, con el nombre del dispositivo al "
                "final",
            ),
            (
                "renombrado durante la edición",
                "un dispositivo renombró el archivo mientras otro seguía editándolo",
                "el archivo editado se guarda con su nombre anterior junto al renombrado",
            ),
        ],
    ),
    (
        ("3307", "3308"),
        "es",
        "El enlace compartido {0}",
        "Quien abre el enlace ve este error porque el enlace {1}. El archivo original no se ve "
        "afectado.",
        "El propietario puede revisar la configuración en Compartir > Enlaces. {2}",
        [
            (
                "caducó",
                "tenía una fecha de caducidad y esa fecha ya pasó",
                "Para recuperar el acceso, pide que amplíe la fecha o que genere un enlace nuevo.",
            ),
            (
                "requiere otra contraseña",
                "está protegido con contraseña y la introducida no es correcta",
                "Para recuperar el acceso, pide la contraseña; tras cinco intentos fallidos el enlace se "
                "bloquea durante 15 minutos.",
            ),
        ],
    ),
    (
        ("6120", "6121"),
        "es",
        "Pago no procesado: {0}",
        "Al renovar la suscripción, la pasarela de pago devolvió este código porque {1}.",
        "Entra en Configuración > Facturación > Medio de pago. {2} El reintento se hace "
        "automáticamente en las siguientes 24 horas.",
        [
            (
                "rechazo del emisor",
                "el banco emisor rechazó la operación sin indicar el motivo",
                "Contacta con tu banco para autorizar las operaciones de Nimbus o registra otra tarjeta.",
            ),
            (
                "tarjeta vencida",
                "la tarjeta registrada superó su fecha de vencimiento",
                "Actualiza la fecha de vencimiento o registra una tarjeta nueva.",
            ),
        ],
    ),
    (
        ("7002", "7003"),
        "en",
        "SAML sign-in failed: {0}",
        "Nimbus rejected the SAML response because {1}. Users who sign in with SSO cannot reach "
        "their files until it is fixed; administrators can still use their password.",
        "Go to Admin console > Security > Single sign-on. {2}, save, and ask one user to sign in "
        "again to confirm.",
        [
            (
                "certificate expired",
                "the signing certificate uploaded for your identity provider has expired",
                "Upload the renewed certificate from your identity provider",
            ),
            (
                "audience mismatch",
                "the Audience value in the SAML response does not match the Entity ID Nimbus expects",
                "Copy the Entity ID shown on that page into the Audience field of your identity "
                "provider",
            ),
        ],
    ),
    (
        ("1404", "1405"),
        "es",
        "Ubicación de la carpeta no compatible ({0})",
        "El cliente de escritorio no puede vigilar los cambios de la carpeta de Nimbus porque está "
        "{1}.",
        "Abre Preferencias > Cuenta > Mover carpeta y elige una ubicación en {2}. Los archivos se "
        "trasladan sin volver a descargarse.",
        [
            (
                "unidad de red",
                "en una unidad de red o en una carpeta compartida por SMB",
                "un disco local",
            ),
            (
                "unidad extraíble",
                "en una memoria USB o en un disco externo que se desconectó",
                "un disco interno que esté siempre conectado",
            ),
        ],
    ),
    (
        ("8010", "8011"),
        "en",
        "Webhook delivery failed: {0}",
        "Nimbus could not deliver an event because {1}. The event is retried with exponential "
        "backoff for up to 24 hours.",
        "{2}; once the endpoint recovers, pending events are delivered in order. You can replay "
        "failed deliveries from Developers > Webhooks.",
        [
            (
                "non-2xx response",
                "your endpoint answered with a status code outside the 2xx range",
                "Check your endpoint logs for the rejected request",
            ),
            (
                "timeout",
                "your endpoint did not answer within 10 seconds",
                "Acknowledge the request right away and do the heavy work asynchronously",
            ),
        ],
    ),
    (
        ("2250", "2251"),
        "es",
        "Nombre de archivo no válido ({0})",
        "El archivo no se sincroniza porque su nombre {1}. El resto de la carpeta sigue "
        "sincronizando con normalidad.",
        "{2}; el cliente lo detecta y lo sube en el siguiente ciclo.",
        [
            (
                "caracteres",
                'contiene caracteres que Nimbus no admite, como < > : " | ? *',
                "Renombra el archivo quitando esos caracteres",
            ),
            (
                "ruta demasiado larga",
                "forma una ruta de más de 400 caracteres contando las carpetas",
                "Acorta el nombre o mueve el archivo a una carpeta menos profunda",
            ),
        ],
    ),
    (
        ("4510", "4511"),
        "es",
        "Código de verificación rechazado ({0})",
        "Al iniciar sesión con verificación en dos pasos, el código de seis dígitos {1}.",
        "{2} La aplicación genera un código nuevo cada 30 segundos.",
        [
            (
                "incorrecto",
                "no coincide con el que espera el servidor; lo más habitual es que la hora del teléfono "
                "esté desajustada",
                "Activa la hora automática en el teléfono y escribe el código nuevo que muestre la "
                "aplicación de autenticación.",
            ),
            (
                "ya utilizado",
                "ya se usó en un inicio de sesión anterior y cada código vale una sola vez",
                "Espera a que la aplicación de autenticación genere el siguiente código y escríbelo.",
            ),
        ],
    ),
    (
        ("9001", "9002"),
        "en",
        "Data export unavailable: {0}",
        "The download page for your data export shows this code because {1}.",
        "{2} from Settings > Privacy > Export your data. Only one export can run at a time per "
        "account.",
        [
            (
                "still being prepared",
                "the archive is still being prepared; large accounts can take up to 48 hours",
                "Wait for the email confirming the archive is ready, then download it",
            ),
            (
                "link expired",
                "the download link expired; links stay valid for 7 days after the archive is ready",
                "Request a new export",
            ),
        ],
    ),
    (
        ("3600", "3601"),
        "es",
        "No se puede restaurar la versión ({0})",
        "Intentaste restaurar una versión anterior de un archivo que {1}.",
        "Abre el historial del archivo para ver qué versiones están disponibles. {2}",
        [
            (
                "fuera del historial",
                "es más antigua que el período de historial de versiones de tu plan",
                "Las versiones fuera de ese período no se pueden recuperar; el período depende del plan "
                "contratado.",
            ),
            (
                "retención legal",
                "está bloqueada por una retención legal activa sobre la carpeta",
                "Solo el administrador que creó la retención legal puede levantarla; mientras tanto la "
                "versión se puede descargar, pero no restaurar.",
            ),
        ],
    ),
    # singletons
    (
        ("1001",),
        "es",
        "Versión del cliente no compatible",
        "La aplicación de escritorio o móvil es anterior a la versión mínima que admite el "
        "servidor, y la conexión se rechaza para evitar daños en la sincronización.",
        "Descarga la última versión desde la página de descargas e instálala encima de la actual; "
        "no hace falta desinstalar antes ni volver a vincular la cuenta.",
        [()],
    ),
    (
        ("1720",),
        "es",
        "Reloj del sistema desajustado",
        "La hora del equipo difiere en más de cinco minutos de la del servidor, y las firmas de las "
        "peticiones se consideran inválidas.",
        "Activa la sincronización automática de fecha y hora en el sistema operativo y reinicia la "
        "aplicación.",
        [()],
    ),
    (
        ("2999",),
        "en",
        "Encryption key missing",
        "This folder uses end-to-end encryption and the device does not hold the key needed to "
        "decrypt it, usually because the device was set up after the folder was created.",
        "Approve the new device from one that already has access (Settings > Security > Trusted "
        "devices), or enter your recovery phrase. Nimbus support cannot recover the key for you.",
        [()],
    ),
    (
        ("3150",),
        "es",
        "Archivo bloqueado por otra persona",
        "Otra persona del equipo tiene el archivo abierto en modo de edición exclusiva, y Nimbus no "
        "acepta cambios de nadie más mientras dure el bloqueo.",
        "Puedes abrirlo en modo lectura o pedir que lo libere. Los bloqueos caducan a las 8 horas y "
        "un administrador puede forzar la liberación desde el panel del archivo.",
        [()],
    ),
    (
        ("4600",),
        "es",
        "Región de datos no disponible",
        "La organización está configurada para guardar los datos en una región que su plan no "
        "incluye, por ejemplo Australia con un plan limitado a Europa y Norteamérica.",
        "Cambia la región en Consola > Organización > Ubicación de los datos o contrata un plan "
        "con residencia de datos ampliada.",
        [()],
    ),
    (
        ("5500",),
        "es",
        "Servicio en mantenimiento",
        "Una parte del servicio está en mantenimiento programado y las operaciones de escritura se "
        "pausan durante la ventana.",
        "No hace falta hacer nada: los cambios quedan en cola y se sincronizan al terminar. Las "
        "ventanas de mantenimiento se anuncian en la página de estado.",
        [()],
    ),
    (
        ("6400",),
        "es",
        "Datos fiscales incompletos",
        "No se pudo emitir la factura porque falta el identificador fiscal o la dirección de "
        "facturación está incompleta.",
        "Completa los campos en Configuración > Facturación > Datos fiscales. La factura se emite "
        "de nuevo en unos minutos y llega al correo de facturación.",
        [()],
    ),
    (
        ("7700",),
        "en",
        "Invalid API key",
        "The request carried an API key that does not exist, was deleted, or belongs to another "
        "organization.",
        "Create a new key in Developers > API keys and update your integration. Keys are shown "
        "only once, so keep them in a secret manager.",
        [()],
    ),
    (
        ("8800",),
        "es",
        "Límite de dispositivos alcanzado",
        "La cuenta ya tiene vinculado el número máximo de dispositivos que permite el plan, y el "
        "nuevo no puede iniciar la sincronización.",
        "Desvincula un dispositivo que ya no uses desde Configuración > Dispositivos y vuelve a "
        "intentarlo en el nuevo.",
        [()],
    ),
    (
        ("9300",),
        "en",
        "File quarantined",
        "The malware scan flagged the file when it was uploaded, so it is kept in quarantine and "
        "cannot be shared or downloaded.",
        "If you believe the file is safe, its owner can request a manual review from the file's "
        "details panel. Reviews take up to two business days.",
        [()],
    ),
]

for codes, lang, title, cause, fix, slots in ERRORS:
    h_cause, h_fix = rng.choice(ERROR_HEADINGS[lang])
    for code, s in zip(codes, slots, strict=True):
        err = f"ERR-{code}"
        text = (
            f"{err} · {title.format(*s)}\n\n"
            f"{h_cause}: {cause.format(*s)}\n\n"
            f"{h_fix}: {fix.format(*s)}"
        )
        add(f"err-{code}", text, "error_article", "exact", lang)

# ----------------------------------------------------------------------------- plans / SKUs
# Siblings (monthly/annual, small/large) share one template; only the limits and price differ.

SKU_TEMPLATES = {
    "es": [
        "{code} — {name}\n\n{desc}\n\nIncluye: {storage}; {users}; {history}.\nPrecio: {price}.",
        "{name} ({code})\n\n{desc} Límites: {storage}, {users} y {history}.\n\nPrecio de "
        "lista: {price}.",
        "Ficha de producto {code}: {name}\n\n{desc}\n\n- Almacenamiento: {storage}\n- Usuarios: "
        "{users}\n- Historial: {history}\n- Precio: {price}",
    ],
    "en": [
        "{code} — {name}\n\n{desc}\n\nIncludes: {storage}; {users}; {history}.\nPrice: {price}.",
        "Product sheet {code}: {name}\n\n{desc}\n\n- Storage: {storage}\n- Users: {users}\n- "
        "History: {history}\n- Price: {price}",
    ],
}

SKUS = [
    (
        "es",
        [
            (
                "10A",
                "Nimbus Free",
                "Plan gratuito para uso personal. Sin soporte por chat; la "
                "ayuda llega por el centro de ayuda y la comunidad.",
                "15 GB",
                "1 usuario",
                "30 días de historial de versiones",
                "gratis",
            )
        ],
    ),
    (
        "es",
        [
            (
                "21M",
                "Nimbus Personal, facturación mensual",
                "Plan individual con sincronización "
                "en todos tus dispositivos; se factura cada mes.",
                "2 TB",
                "1 usuario",
                "90 días de historial de versiones",
                "8 USD al mes",
            ),
            (
                "21A",
                "Nimbus Personal, facturación anual",
                "Plan individual con sincronización "
                "en todos tus dispositivos; se factura una vez al año.",
                "2 TB",
                "1 usuario",
                "90 días de historial de versiones",
                "80 USD al año",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "34M",
                "Nimbus Familia, facturación mensual",
                "Espacio compartido para un hogar, "
                "con carpetas privadas para cada miembro; se factura cada mes.",
                "2 TB compartidos",
                "hasta 6 miembros",
                "90 días de historial de versiones",
                "12 USD al mes",
            ),
            (
                "34A",
                "Nimbus Familia, facturación anual",
                "Espacio compartido para un hogar, "
                "con carpetas privadas para cada miembro; se factura una vez al año.",
                "2 TB compartidos",
                "hasta 6 miembros",
                "90 días de historial de versiones",
                "120 USD al año",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "52M",
                "Nimbus Team, facturación mensual",
                "Para equipos pequeños: consola de "
                "administración, carpetas de equipo y registro de actividad; se factura cada mes.",
                "1 TB por usuario",
                "mínimo 3 usuarios",
                "180 días de historial de versiones",
                "9 USD por usuario al mes",
            ),
            (
                "52A",
                "Nimbus Team, facturación anual",
                "Para equipos pequeños: consola de "
                "administración, carpetas de equipo y registro de actividad; se factura una vez al "
                "año.",
                "1 TB por usuario",
                "mínimo 3 usuarios",
                "180 días de historial de versiones",
                "90 USD por usuario al año",
            ),
        ],
    ),
    (
        "en",
        [
            (
                "77A",
                "Nimbus Business, annual billing",
                "For growing companies: everything in "
                "Team plus SAML single sign-on and data residency in Europe or North America; "
                "billed once a year.",
                "5 TB per user",
                "minimum 3 users",
                "365 days of version history",
                "200 USD per user per year",
            ),
            (
                "77M",
                "Nimbus Business, monthly billing",
                "For growing companies: everything in "
                "Team plus SAML single sign-on and data residency in Europe or North America; "
                "billed every month.",
                "5 TB per user",
                "minimum 3 users",
                "365 days of version history",
                "20 USD per user per month",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "90E",
                "Nimbus Enterprise",
                "Para organizaciones grandes: retención legal, "
                "residencia de datos en cualquier región y gestor de cuenta dedicado.",
                "capacidad a medida",
                "desde 50 usuarios",
                "historial de versiones ilimitado",
                "a cotizar con el equipo comercial",
            )
        ],
    ),
    (
        "es",
        [
            (
                "15S",
                "Ampliación de almacenamiento 1 TB",
                "Complemento que suma capacidad a cualquier plan de pago, personal o de equipo.",
                "1 TB adicional",
                "se comparte entre los usuarios del plan",
                "mismo historial que el plan base",
                "5 USD al mes",
            ),
            (
                "15T",
                "Ampliación de almacenamiento 10 TB",
                "Complemento que suma capacidad a cualquier plan de pago, personal o de equipo.",
                "10 TB adicionales",
                "se comparte entre los usuarios del plan",
                "mismo historial que el plan base",
                "40 USD al mes",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "40B",
                "Nimbus Backup para equipos de escritorio",
                "Copia de seguridad automática de un ordenador completo, con copias cada hora.",
                "sin límite de espacio",
                "1 equipo por licencia",
                "1 año de copias conservadas",
                "4 USD al mes por equipo",
            ),
            (
                "41B",
                "Nimbus Backup para servidores",
                "Copia de seguridad automática de un "
                "servidor completo, con copias cada 15 minutos.",
                "sin límite de espacio",
                "1 servidor por licencia",
                "1 año de copias conservadas",
                "15 USD al mes por servidor",
            ),
        ],
    ),
    (
        "en",
        [
            (
                "26X",
                "Nimbus Transfer",
                "Send large files to people without a Nimbus account; "
                "each transfer can be up to 250 GB and its link lasts 14 days.",
                "no storage counted against your plan",
                "1 sender",
                "transfers are not versioned",
                "12 USD per month",
            ),
            (
                "26Y",
                "Nimbus Transfer Plus",
                "Send large files to people without a Nimbus "
                "account; each transfer can be up to 1 TB and its link lasts 60 days.",
                "no storage counted against your plan",
                "1 sender",
                "transfers are not versioned",
                "25 USD per month",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "63E",
                "Nimbus Team para Educación",
                "Plan Team con precio especial para instituciones educativas acreditadas.",
                "1 TB por usuario",
                "mínimo 10 usuarios",
                "180 días de historial de versiones",
                "3 USD por usuario al mes",
            ),
            (
                "63N",
                "Nimbus Team para ONG",
                "Plan Team con precio especial para organizaciones sin fines de lucro registradas.",
                "1 TB por usuario",
                "mínimo 10 usuarios",
                "180 días de historial de versiones",
                "4 USD por usuario al mes",
            ),
        ],
    ),
    (
        "es",
        [
            (
                "88R",
                "Retención avanzada (complemento)",
                "Conserva los archivos eliminados y sus "
                "versiones durante el plazo que fije el administrador, hasta 10 años.",
                "no consume cuota",
                "se contrata para todos los usuarios del plan",
                "hasta 10 años de conservación",
                "3 USD por usuario al mes",
            ),
            (
                "88L",
                "Retención legal (complemento)",
                "Bloquea carpetas completas ante un litigio "
                "o una auditoría, sin que nadie pueda modificar ni eliminar su contenido.",
                "no consume cuota",
                "se contrata para todos los usuarios del plan",
                "conservación indefinida mientras dure la retención",
                "6 USD por usuario al mes",
            ),
        ],
    ),
]

for lang, group in SKUS:
    tpl = rng.choice(SKU_TEMPLATES[lang])
    for code, name, desc, storage, users, history, price in group:
        sku = f"SKU-{code}"
        text = tpl.format(
            code=sku,
            name=name,
            desc=desc,
            storage=storage,
            users=users,
            history=history,
            price=price,
        )
        add(f"sku-{code.lower()}", text, "plan", "exact", lang)

# ------------------------------------------------------------------------------ billing FAQs
# Formal support vocabulary on purpose. Colloquial questions ("me cobraron dos veces") must find
# these through meaning, not shared words.

FAQ_FORMATS = {
    "es": ["{q}\n\n{a}", "Pregunta frecuente: {q}\n\n{a}", "{q}\n\nRespuesta: {a}"],
    "en": ["{q}\n\n{a}", "FAQ: {q}\n\n{a}"],
}
FAQ_CLOSINGS = {
    "es": [
        "",
        "",
        "Si la situación persiste, el equipo de facturación atiende desde el centro de ayuda.",
        "Este procedimiento aplica a todos los planes de pago.",
        "Los administradores de equipo encuentran la misma opción en la consola.",
    ],
    "en": ["", "", "If the issue persists, contact the billing team from the help center."],
}

FAQS = [
    (
        "cargo-duplicado",
        "es",
        "¿Qué ocurre si aparece un cargo duplicado en el medio de pago?",
        "Cuando el resumen de la tarjeta refleja un cargo duplicado correspondiente al mismo período "
        "de facturación, la segunda operación suele ser una preautorización que el emisor libera "
        "en un plazo de 3 a 5 días hábiles. Si transcurrido ese plazo el importe persiste, solicita "
        "la anulación desde Facturación > Movimientos indicando la fecha de ambas operaciones; el "
        "importe se reintegra al mismo medio de pago.",
    ),
    (
        "reembolso-prorrateado",
        "es",
        "¿Cómo se calcula el reembolso prorrateado al pasar a un plan inferior?",
        "Al cambiar a un plan de menor categoría, el importe correspondiente a los días restantes "
        "del ciclo vigente se calcula de forma proporcional y se acredita como saldo a favor para "
        "las siguientes facturas. El nuevo plan entra en vigor de inmediato. Las condiciones de "
        "devolución en efectivo se rigen por la Política de reembolsos.",
    ),
    (
        "baja-suscripcion",
        "es",
        "¿Cómo se tramita la baja de la suscripción?",
        "La baja de la suscripción se tramita en Configuración > Plan > Finalizar suscripción. El "
        "servicio contratado se mantiene hasta el término del ciclo ya abonado y, a partir de esa "
        "fecha, la cuenta pasa a la modalidad gratuita. No se emiten nuevas facturas después de "
        "confirmar la operación.",
    ),
    (
        "medio-de-pago",
        "es",
        "¿Cómo se modifica el medio de pago registrado?",
        "El titular de la cuenta puede sustituir la tarjeta o la cuenta bancaria asociada en "
        "Configuración > Facturación > Medio de pago. El nuevo medio se utiliza a partir de la "
        "siguiente renovación; las facturas ya emitidas no se modifican.",
    ),
    (
        "datos-fiscales",
        "es",
        "¿Cómo se incorporan los datos fiscales a las facturas?",
        "En Configuración > Facturación > Datos fiscales se registran la razón social, el "
        "identificador tributario y el domicilio fiscal. Los datos se aplican a las facturas que se "
        "emitan desde ese momento; para rectificar una factura anterior es necesario solicitar una "
        "nota de crédito.",
    ),
    (
        "historial-facturacion",
        "es",
        "¿Dónde se consultan los comprobantes emitidos?",
        "El historial de facturación, con todos los comprobantes en formato PDF, se encuentra en "
        "Configuración > Facturación > Comprobantes. Cada documento puede obtenerse de forma "
        "individual o en un archivo comprimido por ejercicio.",
    ),
    (
        "renovacion-automatica",
        "es",
        "¿Cómo funciona la renovación automática?",
        "Las suscripciones se renuevan de forma automática al final de cada ciclo con el mismo plan "
        "y el mismo medio de pago. El sistema envía un aviso por correo electrónico siete días "
        "antes de la renovación de los ciclos anuales. La renovación automática puede "
        "desactivarse en Configuración > Plan.",
    ),
    (
        "impuestos-moneda",
        "es",
        "¿En qué moneda se emiten los importes y qué impuestos se aplican?",
        "Los importes se expresan en dólares estadounidenses. Según el país del domicilio fiscal, "
        "se adicionan el IVA u otros tributos locales, que figuran discriminados en el comprobante. "
        "La conversión a la moneda local la realiza la entidad emisora de la tarjeta.",
    ),
    (
        "periodo-prueba",
        "es",
        "¿Qué sucede al finalizar el período de prueba?",
        "El período de evaluación de los planes de equipo dura 14 días. Al concluir, la "
        "suscripción se activa con el medio de pago informado, salvo que el administrador la haya "
        "finalizado previamente. Si no se registró ningún medio de pago, el equipo pasa a modo de "
        "solo lectura.",
    ),
    (
        "periodo-gracia",
        "es",
        "¿Existe un período de gracia ante un pago no acreditado?",
        "Cuando la renovación no puede acreditarse, la cuenta conserva todas sus funciones durante "
        "un período de gracia de 10 días, en el que se efectúan reintentos automáticos. El titular "
        "recibe una notificación con cada reintento fallido.",
    ),
    (
        "suspension-impago",
        "es",
        "¿En qué casos se suspende una cuenta por falta de pago?",
        "Concluido el período de gracia sin que se regularice el saldo, la cuenta queda suspendida: "
        "los archivos se conservan y pueden descargarse, pero se inhabilitan la sincronización y la "
        "colaboración. La suspensión no implica la eliminación de los datos.",
    ),
    (
        "reactivacion-cuenta",
        "es",
        "¿Cómo se reactiva una cuenta suspendida?",
        "Para restablecer una cuenta suspendida es necesario regularizar el saldo pendiente desde "
        "Configuración > Facturación. La reactivación es inmediata una vez acreditado el importe y "
        "la sincronización se reanuda sin pérdida de información.",
    ),
    (
        "transferencia-titularidad",
        "es",
        "¿Cómo se transfiere la titularidad de una cuenta de equipo?",
        "El titular puede designar a otro administrador como nuevo titular en Consola > "
        "Organización > Titularidad. La transferencia requiere la aceptación expresa del "
        "destinatario y traslada la responsabilidad de facturación a partir del ciclo siguiente.",
    ),
    (
        "eliminacion-cuenta",
        "es",
        "¿Cómo se solicita la eliminación definitiva de la cuenta?",
        "La supresión definitiva se solicita en Configuración > Privacidad > Eliminar cuenta. "
        "Previamente debe finalizarse cualquier suscripción vigente. Una vez confirmada, la "
        "operación es irreversible y el contenido deja de ser recuperable.",
    ),
    (
        "cambio-correo",
        "es",
        "¿Cómo se actualiza la dirección de correo electrónico de la cuenta?",
        "La dirección asociada se modifica en Configuración > Perfil. El sistema envía un enlace de "
        "confirmación a la nueva dirección y una notificación a la anterior; el cambio se hace "
        "efectivo al validar el enlace.",
    ),
    (
        "restablecer-contrasena",
        "es",
        "¿Cómo se restablece la contraseña de acceso?",
        "Desde la pantalla de acceso, la opción «Restablecer contraseña» envía un enlace de un solo "
        "uso, válido por 60 minutos, a la dirección registrada. Por seguridad, al completar el "
        "restablecimiento se cierran las sesiones abiertas en los demás dispositivos.",
    ),
    (
        "activar-dos-pasos",
        "es",
        "¿Cómo se habilita la verificación en dos pasos?",
        "En Configuración > Seguridad > Verificación en dos pasos se vincula una aplicación de "
        "autenticación mediante el código QR. Al finalizar, el sistema genera diez códigos de "
        "respaldo que conviene resguardar fuera del dispositivo.",
    ),
    (
        "perdida-segundo-factor",
        "es",
        "¿Qué procedimiento corresponde ante la pérdida del segundo factor?",
        "Si el dispositivo de autenticación no está disponible, el ingreso puede completarse con "
        "uno de los códigos de respaldo. En ausencia de estos, el soporte inicia un proceso de "
        "verificación de identidad, que exige acreditar la titularidad de la cuenta.",
    ),
    (
        "capacidad-adicional",
        "es",
        "¿Cómo se incorpora capacidad de almacenamiento adicional?",
        "Los planes de pago admiten complementos de capacidad que se contratan desde Configuración "
        "> Plan > Complementos. La capacidad se habilita de inmediato y su importe se prorratea "
        "hasta la siguiente renovación.",
    ),
    (
        "descuento-anual",
        "es",
        "¿Qué bonificación ofrece la modalidad anual?",
        "La contratación anual equivale a diez mensualidades: se abonan 10 meses y se obtiene el "
        "servicio durante 12. La bonificación se aplica automáticamente al seleccionar la "
        "periodicidad anual.",
    ),
    (
        "descuento-educativo",
        "es",
        "¿Qué requisitos tiene la bonificación para instituciones educativas?",
        "Las instituciones educativas acceden a una tarifa especial previa acreditación de su "
        "condición mediante documentación oficial. La validación demora hasta cinco días hábiles y "
        "la tarifa se aplica desde la siguiente factura.",
    ),
    (
        "migracion-a-equipo",
        "es",
        "¿Cómo se migra de un plan individual a un plan de equipo?",
        "La migración se realiza desde Configuración > Plan > Pasar a Team. Los archivos de la "
        "cuenta individual se conservan en el espacio personal del titular, y el saldo no "
        "consumido del plan anterior se acredita en la primera factura del plan de equipo.",
    ),
    (
        "alta-usuarios",
        "es",
        "¿Cómo se factura la incorporación de nuevos integrantes al equipo?",
        "Cada integrante incorporado ocupa una licencia. Las licencias añadidas a mitad de ciclo "
        "se facturan de forma proporcional a los días restantes, y a partir de la renovación se "
        "facturan a tarifa completa.",
    ),
    (
        "licencias-no-asignadas",
        "es",
        "¿Se facturan las licencias que no están asignadas?",
        "Sí. La facturación se calcula sobre la cantidad de licencias contratadas, con "
        "independencia de su asignación. Para reducir el importe, el administrador debe disminuir "
        "la cantidad de licencias en Consola > Licencias antes de la renovación.",
    ),
    (
        "baja-integrantes",
        "es",
        "¿Qué sucede con los archivos de un integrante dado de baja del equipo?",
        "Al retirar a un integrante, sus archivos de equipo permanecen en las carpetas compartidas "
        "y sus archivos personales se transfieren al administrador que efectúa la operación. La "
        "licencia queda disponible para ser asignada a otra persona.",
    ),
    (
        "cupon-promocional",
        "es",
        "¿Cómo se aplica un código promocional?",
        "Los códigos promocionales se ingresan en el paso de confirmación de la compra o en "
        "Configuración > Facturación > Promociones. Cada código admite un único uso por cuenta y no "
        "es acumulable con otras bonificaciones.",
    ),
    (
        "contracargo",
        "es",
        "¿Qué consecuencias tiene la apertura de un contracargo?",
        "Cuando el titular de la tarjeta inicia un contracargo ante su entidad, la cuenta queda "
        "suspendida preventivamente hasta la resolución del caso. Se recomienda contactar primero "
        "al equipo de facturación, que puede resolver la mayoría de las discrepancias sin "
        "intervención bancaria.",
    ),
    (
        "cargo-no-reconocido",
        "es",
        "¿Cómo se informa un cargo no reconocido?",
        "Si figura en el resumen un importe atribuido a Nimbus que el titular no reconoce, debe "
        "informarse mediante el formulario de transacciones no reconocidas, indicando los últimos "
        "cuatro dígitos de la tarjeta. El equipo de seguridad verifica si corresponde a otra "
        "cuenta y, en caso de uso indebido, procede a su anulación.",
    ),
    (
        "transferencia-bancaria",
        "es",
        "¿Se admiten pagos por transferencia bancaria u orden de compra?",
        "Los planes Business con contratación anual y el plan Enterprise admiten el pago por "
        "transferencia bancaria contra factura, con un plazo de 30 días. Para emitir una orden de "
        "compra, el área de compras debe solicitar una cotización formal al equipo comercial.",
    ),
    (
        "cambio-ciclo",
        "es",
        "¿Cómo se modifica la periodicidad de facturación?",
        "El paso de periodicidad mensual a anual se realiza en Configuración > Plan > Periodicidad "
        "y se hace efectivo de inmediato, con acreditación proporcional del mes en curso. El cambio "
        "inverso se aplica al término del ciclo anual vigente.",
    ),
    (
        "fecha-de-corte",
        "es",
        "¿Qué fecha determina el inicio de cada ciclo de facturación?",
        "El ciclo comienza el día en que se contrató el plan y se repite en esa misma fecha cada "
        "mes o cada año. En los meses que no contienen dicha fecha, la renovación se efectúa el "
        "último día del mes.",
    ),
    (
        "contacto-facturacion",
        "es",
        "¿Es posible designar un destinatario distinto para las facturas?",
        "En Configuración > Facturación > Contacto puede registrarse una dirección de correo "
        "adicional, por ejemplo la del área contable, que recibirá los comprobantes y los avisos "
        "de renovación sin necesidad de disponer de acceso a la cuenta.",
    ),
    (
        "miembros-familia",
        "es",
        "¿Cómo se administran los integrantes de un plan Familia?",
        "El organizador del plan Familia invita a los integrantes desde Configuración > Familia. "
        "Cada integrante conserva una carpeta privada a la que el organizador no tiene acceso, y "
        "puede retirarse del grupo en cualquier momento.",
    ),
    (
        "cierre-sesiones-remoto",
        "es",
        "¿Cómo se finalizan las sesiones abiertas en dispositivos no reconocidos?",
        "En Configuración > Seguridad > Sesiones activas figuran todos los dispositivos con acceso, "
        "junto con su ubicación aproximada. La opción «Finalizar sesión» revoca el acceso de forma "
        "inmediata; se recomienda además actualizar la contraseña.",
    ),
    (
        "dispositivos-vinculados",
        "es",
        "¿Cuántos dispositivos pueden vincularse a una cuenta?",
        "El plan Free admite hasta tres dispositivos vinculados; los planes de pago no establecen "
        "un máximo. Los dispositivos que no se conectan durante 180 días se desvinculan "
        "automáticamente.",
    ),
    (
        "portabilidad-datos",
        "es",
        "¿Cómo se ejerce el derecho a la portabilidad de los datos?",
        "La exportación de los datos personales se solicita en Configuración > Privacidad > "
        "Exportar datos. El sistema genera un archivo con los metadatos de la cuenta, el "
        "historial de actividad y el contenido almacenado, y notifica por correo cuando está "
        "disponible.",
    ),
    (
        "exceso-cuota",
        "es",
        "¿Qué implica superar la cuota de almacenamiento contratada?",
        "Al alcanzar el límite de capacidad, el contenido existente permanece accesible, pero se "
        "inhabilita la incorporación de nuevos archivos y versiones. No se aplican cargos "
        "adicionales automáticos por exceso; corresponde al titular ampliar la capacidad o depurar "
        "contenido.",
    ),
    (
        "cambio-plan-superior",
        "es",
        "¿Cómo se accede a un plan de categoría superior?",
        "La mejora de plan se realiza desde Configuración > Plan > Cambiar plan. Las nuevas "
        "prestaciones se habilitan de inmediato y se factura únicamente la diferencia proporcional "
        "correspondiente a los días restantes del ciclo.",
    ),
    (
        "cambio-razon-social",
        "es",
        "¿Cómo se modifica la razón social de la organización?",
        "La modificación de la denominación de la organización se efectúa en Consola > "
        "Organización > Datos generales. Para que el cambio se refleje en los comprobantes, deben "
        "actualizarse también los datos fiscales.",
    ),
    (
        "verificacion-identidad",
        "es",
        "¿Qué documentación exige la verificación de identidad?",
        "Para acreditar la titularidad se solicitan los últimos cuatro dígitos del medio de pago "
        "registrado, la fecha aproximada de alta y, en su caso, el número de una factura emitida. "
        "La documentación se remite exclusivamente a través del formulario seguro del centro de "
        "ayuda.",
    ),
    (
        "upgrade-proration",
        "en",
        "How is a mid-cycle upgrade billed?",
        "When an account moves to a higher tier during a billing period, the new features become "
        "available immediately and the invoice includes only the prorated difference for the "
        "remaining days of the current term.",
    ),
    (
        "billing-address",
        "en",
        "How do I update the billing address on invoices?",
        "The billing address is maintained under Settings > Billing > Tax details. Changes apply "
        "to invoices issued after the update; previously issued invoices remain unchanged and can "
        "be corrected only through a credit note.",
    ),
    (
        "tax-exemption",
        "en",
        "How can an organization apply a tax exemption?",
        "Organizations eligible for tax exemption can upload a valid exemption certificate under "
        "Settings > Billing > Tax details. Once the certificate is verified, subsequent invoices "
        "are issued without the applicable sales tax.",
    ),
    (
        "invoice-delivery",
        "en",
        "Can invoices be delivered to an accounts payable mailbox?",
        "Yes. An additional billing contact can be registered under Settings > Billing > Contacts. "
        "That address receives every invoice and renewal notice without needing a Nimbus seat.",
    ),
    (
        "sso-enforcement",
        "en",
        "What happens to password sign-in when SSO is enforced?",
        "Once an administrator enforces single sign-on, members can no longer authenticate with a "
        "Nimbus password and must use the identity provider. Administrators retain password access "
        "so the organization is never locked out.",
    ),
    (
        "deleted-account-recovery",
        "en",
        "Can a deleted account be restored?",
        "An account scheduled for deletion enters a 14-day grace period during which the owner can "
        "cancel the request by signing in. After that period, deletion is irreversible and "
        "content cannot be retrieved.",
    ),
    (
        "seat-true-up",
        "en",
        "How are additional seats reconciled on annual Team subscriptions?",
        "Seats added during an annual term are reconciled quarterly: the account is invoiced for "
        "the added seats, prorated for the remainder of the term, at the start of each quarter.",
    ),
    (
        "charge-currency",
        "en",
        "Why can the charged total differ from the list price?",
        "Nimbus invoices in US dollars. When the payment instrument is denominated in another "
        "currency, the issuer applies its own exchange rate and may add a foreign transaction fee, "
        "which is outside Nimbus's control.",
    ),
    (
        "free-tier-downgrade",
        "en",
        "What happens to content when moving to the free tier?",
        "Content above the free storage limit stays accessible for reading and downloading, but "
        "new uploads are disabled until usage falls below the limit. Shared team folders are "
        "transferred to the remaining administrator.",
    ),
    (
        "subscription-pause",
        "en",
        "Can a subscription be temporarily suspended?",
        "Subscriptions cannot be suspended temporarily. Account holders who expect a period of "
        "inactivity can move to the free tier and return to a paid plan later; content within the "
        "free storage limit is retained.",
    ),
]

for slug, lang, q, a in FAQS:
    text = rng.choice(FAQ_FORMATS[lang]).format(q=q, a=a)
    closing = rng.choice(FAQ_CLOSINGS[lang])
    add(f"faq-{slug}", f"{text} {closing}".strip(), "faq", "paraphrase", lang)

# ---------------------------------------------------------------------- versioned policies
# Two versions per title. The text is identical except the version line and the facts in
# FACTS; the 2024 version is marked as superseded.

POLICY_VERSION_LINE = {
    "es": {
        "2024": "Versión 2024. Reemplazada por la versión 2026.",
        "2026": "Versión 2026. Vigente.",
    },
    "en": {
        "2024": "Version 2024. Superseded by the 2026 version.",
        "2026": "Version 2026. In force.",
    },
}

POLICIES = [
    (
        "reembolsos",
        "es",
        "Política de reembolsos",
        "Puedes solicitar la devolución completa de una suscripción dentro de los {0} días "
        "siguientes al primer cargo. Este derecho alcanza a {1}. Pasado ese plazo no se devuelven "
        "importes, aunque los cambios de plan generan saldo a favor según el cálculo proporcional.",
        {
            "2024": ("14", "las suscripciones anuales; las mensuales no tienen devolución"),
            "2026": ("30", "las suscripciones anuales y también a las mensuales"),
        },
    ),
    (
        "papelera",
        "es",
        "Política de conservación de la papelera",
        "Los archivos eliminados permanecen en la papelera durante {0} días y después se borran de "
        "forma definitiva. En los planes de equipo el plazo es de {1} días. Vaciar la papelera "
        "manualmente borra su contenido en el acto.",
        {"2024": ("30", "60"), "2026": ("60", "120")},
    ),
    (
        "cuentas-inactivas",
        "es",
        "Política de cuentas inactivas",
        "Una cuenta del plan Free sin inicios de sesión durante {0} meses se considera inactiva y "
        "su contenido se elimina. Enviamos un aviso al correo de la cuenta {1} días antes; basta "
        "con iniciar sesión para conservarla. Las cuentas de pago nunca se consideran inactivas.",
        {"2024": ("12", "14"), "2026": ("24", "30")},
    ),
    (
        "contrasenas",
        "es",
        "Política de contraseñas",
        "Las contraseñas deben tener al menos {0} caracteres y no pueden coincidir con las últimas "
        "cinco utilizadas. La verificación en dos pasos es {1}.",
        {
            "2024": ("8", "recomendada para todas las cuentas, sin ser obligatoria"),
            "2026": ("12", "obligatoria para los administradores y recomendada para el resto"),
        },
    ),
    (
        "uso-aceptable",
        "es",
        "Política de uso aceptable",
        "Nimbus es un servicio de almacenamiento y colaboración, no una red de distribución de "
        "contenidos. Cada cuenta puede descargar hasta {0} por día entre todos sus enlaces "
        "compartidos; al superar ese volumen los enlaces se pausan hasta el día siguiente. El "
        "alojamiento de material ilícito implica la suspensión inmediata.",
        {"2024": ("1 TB",), "2026": ("3 TB",)},
    ),
    (
        "registros-acceso",
        "es",
        "Política de conservación de registros de acceso",
        "Guardamos los registros de acceso (dirección IP, dispositivo y fecha de cada inicio de "
        "sesión) durante {0} días para investigar incidentes de seguridad. Los administradores de "
        "equipo pueden consultarlos y exportarlos durante ese período.",
        {"2024": ("90",), "2026": ("180",)},
    ),
    (
        "soporte",
        "es",
        "Política de tiempos de respuesta del soporte",
        "El tiempo máximo hasta la primera respuesta del soporte depende del plan: {0} para "
        "Personal y Familia, {1} para Team y {2} para Business. Enterprise tiene un acuerdo propio. "
        "Los plazos se cuentan en horas hábiles.",
        {"2024": ("48 horas", "24 horas", "8 horas"), "2026": ("24 horas", "12 horas", "4 horas")},
    ),
    (
        "disponibilidad",
        "es",
        "Política de disponibilidad del servicio",
        "Nimbus se compromete a una disponibilidad mensual del {0} para los planes Business y "
        "Enterprise. Si no se cumple, el cliente recibe un crédito del {1} de la cuota de ese mes, "
        "que debe reclamarse dentro de los 30 días siguientes.",
        {"2024": ("99,9 %", "10 %"), "2026": ("99,95 %", "25 %")},
    ),
    (
        "cambios-precio",
        "es",
        "Política de cambios de precio",
        "Cuando cambian los precios de un plan, avisamos a los clientes afectados con {0} días de "
        "antelación. El nuevo precio se aplica a partir de la primera renovación posterior al "
        "aviso; las suscripciones anuales mantienen su precio hasta el final del ciclo.",
        {"2024": ("30",), "2026": ("60",)},
    ),
    (
        "conservacion-tras-baja",
        "es",
        "Política de conservación de datos tras la baja",
        "Al finalizar una suscripción de pago, los archivos que superan la cuota gratuita se "
        "conservan durante {0} días en modo de solo lectura. Pasado ese plazo se eliminan de "
        "forma definitiva, empezando por los más antiguos.",
        {"2024": ("30",), "2026": ("90",)},
    ),
    (
        "enlaces-compartidos",
        "es",
        "Política de enlaces compartidos",
        "Los enlaces públicos creados sin fecha de caducidad explícita caducan a los {0} días. La "
        "caducidad máxima que puede fijarse es de {1} días; los administradores pueden reducir "
        "ambos valores para su organización.",
        {"2024": ("90", "365"), "2026": ("30", "180")},
    ),
    (
        "edad-minima",
        "es",
        "Política de edad mínima",
        "Para crear una cuenta propia en Nimbus hay que tener al menos {0} años. Por debajo de esa "
        "edad, solo se puede usar Nimbus como integrante de un plan Familia administrado por una "
        "persona adulta.",
        {"2024": ("13",), "2026": ("16",)},
    ),
    (
        "subprocessors",
        "en",
        "Subprocessor change policy",
        "Nimbus notifies customers at least {0} days before engaging a new subprocessor that will "
        "handle customer data. Customers may object during that window; if the objection cannot "
        "be resolved, they may terminate the affected service with a prorated refund.",
        {"2024": ("15",), "2026": ("30",)},
    ),
    (
        "incident-notification",
        "en",
        "Security incident notification policy",
        "If a security incident affects customer data, Nimbus notifies the affected account owners "
        "within {0} hours of confirming it. The notice describes the data involved, the measures "
        "taken and a contact for follow-up questions.",
        {"2024": ("72",), "2026": ("48",)},
    ),
    (
        "api-fair-use",
        "en",
        "API fair use policy",
        "Each API key may send up to {0} requests per minute and {1} requests per day. Requests "
        "above those limits receive HTTP 429. Higher limits are available on request for Business "
        "and Enterprise plans.",
        {"2024": ("600", "100,000"), "2026": ("1,200", "250,000")},
    ),
]

for slug, lang, title, body, facts in POLICIES:
    for version in ("2024", "2026"):
        text = f"{title}\n{POLICY_VERSION_LINE[lang][version]}\n\n{body.format(*facts[version])}"
        add(f"pol-{slug}-{version}", text, "policy", "distractor", lang, version)

# ------------------------------------------------------------------------------------ noise
# Release notes draw bullets from a pool; blog posts are written by hand. Both reuse product
# vocabulary (codes, plans, refunds, two-step verification) without stating any answer.

RN_BULLETS = {
    "es": [
        "Mejoramos el mensaje que acompaña a ERR-4012 para que sea más claro.",
        "El aviso de ERR-5003 ahora incluye un enlace al centro de ayuda.",
        "Rediseñamos la pantalla de facturación y la vista de comprobantes.",
        "La papelera carga más rápido en carpetas con miles de archivos.",
        "Nuevo indicador de progreso al subir archivos grandes.",
        "Corregimos un fallo visual en la pantalla de verificación en dos pasos.",
        "La lista de dispositivos vinculados muestra el sistema operativo de cada uno.",
        "Actualizamos las traducciones de la sección de reembolsos del centro de ayuda.",
        "El historial de versiones admite vista previa de imágenes.",
        "Mejoras de rendimiento en la sincronización de carpetas de equipo.",
        "La consola de administración permite filtrar licencias por estado.",
        "Ajustamos los colores del modo oscuro.",
        "Los enlaces compartidos muestran una vista previa al pegarlos en chats.",
        "Corregimos un caso poco frecuente en el que el cliente mostraba ERR-2101 dos veces.",
        "El panel de cuota de almacenamiento se actualiza en tiempo real.",
        "Ahora puedes ordenar la papelera por fecha de eliminación.",
        "La aplicación recuerda la última carpeta abierta.",
        "Mejoramos la accesibilidad del formulario de cambio de plan.",
        "Corrección de errores menores en la pantalla de suscripción.",
        "El buscador encuentra archivos por el nombre de quien los subió.",
        "Nuevo atajo de teclado para compartir un archivo.",
        "La pantalla de inicio de sesión con SSO carga más rápido.",
        "Actualizamos las bibliotecas internas de cifrado.",
        "Reducimos el consumo de batería en segundo plano.",
        "Los códigos SKU de los complementos aparecen en el resumen de compra.",
    ],
    "en": [
        "Clearer wording for the ERR-7003 sign-in message.",
        "The webhook settings page now shows the last delivery time.",
        "Faster loading of the billing history page.",
        "Fixed a layout glitch in the two-step verification screen.",
        "The device list shows when each device last synced.",
        "Improved upload progress reporting for large files.",
        "Dark mode color adjustments.",
        "Minor bug fixes in the subscription screen.",
        "The data export page shows an estimated preparation time.",
        "Keyboard shortcut to copy a share link.",
        "Reduced memory use while indexing large folders.",
        "Updated translations for the refunds help section.",
    ],
}
RN_HEADER = {
    "es": "Notas de versión — Nimbus para {p} {v}\n\nPublicada el {d}. Novedades:",
    "en": "Release notes — Nimbus for {p} {v}\n\nReleased on {d}. What's new:",
}
PLATFORMS = ["Windows", "macOS", "Linux", "iOS", "Android", "web"]
MONTHS = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]
MONTHS_EN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

for i in range(30):
    lang = "en" if i % 5 == 4 else "es"
    platform = rng.choice(PLATFORMS)
    version = f"{rng.randint(5, 7)}.{rng.randint(0, 9)}.{rng.randint(0, 4)}"
    month, day, year = rng.randrange(12), rng.randint(1, 28), rng.choice([2024, 2025])
    date = (
        f"{day} de {MONTHS[month]} de {year}"
        if lang == "es"
        else f"{MONTHS_EN[month]} {day}, {year}"
    )
    bullets = rng.sample(RN_BULLETS[lang], rng.randint(3, 5))
    text = (
        RN_HEADER[lang].format(p=platform, v=version, d=date)
        + "\n"
        + "\n".join(f"- {b}" for b in bullets)
    )
    add(f"rn-{i + 1:02d}", text, "release_notes", "noise", lang)

BLOG = [
    (
        "es",
        "Cómo ordenamos nuestras propias carpetas de equipo",
        "En Nimbus usamos Nimbus para todo, y eso incluye discutir cómo nombrar carpetas. Después "
        "de varios intentos llegamos a una regla simple: una carpeta por proyecto, una subcarpeta "
        "por año y nada de archivos sueltos en la raíz. No es la única forma, pero a nosotros nos "
        "ahorró muchas búsquedas.",
    ),
    (
        "es",
        "Lo que aprendimos rediseñando la pantalla de facturación",
        "La pantalla de facturación era la más visitada y la peor valorada. Entrevistamos a "
        "clientes de los planes Team y Business, y casi todos querían lo mismo: ver de un vistazo "
        "qué pagan y cuándo. Este artículo cuenta el proceso de diseño, no los precios.",
    ),
    (
        "es",
        "Por qué los reembolsos son un tema de producto",
        "Solemos pensar en los reembolsos como un asunto de finanzas, pero cada solicitud cuenta "
        "una historia: una expectativa que el producto no cumplió. Revisamos un trimestre de "
        "solicitudes para entender qué estaba fallando en la experiencia de compra.",
    ),
    (
        "es",
        "Historias de clientes: un estudio de arquitectura en la nube",
        "Un estudio de arquitectura de Valparaíso movió veinte años de planos a Nimbus Business. "
        "Hablamos con su responsable de sistemas sobre la migración, la convivencia con archivos "
        "enormes y lo que cambió en la forma de trabajar del equipo.",
    ),
    (
        "es",
        "La verificación en dos pasos, explicada a tu familia",
        "Explicar la verificación en dos pasos en una cena familiar no es fácil. Probamos varias "
        "metáforas, desde la llave y el candado hasta el portero del edificio, y elegimos las que "
        "mejor funcionaron. Ideal para compartir con quien siempre pregunta para qué sirve.",
    ),
    (
        "es",
        "Detrás de escena: así funciona nuestro equipo de soporte",
        "Nuestro equipo de soporte atiende en seis idiomas desde cuatro países. En este artículo "
        "cuentan cómo organizan los turnos, qué herramientas usan y qué consultas reciben con más "
        "frecuencia en cada época del año.",
    ),
    (
        "es",
        "Diez años de Nimbus",
        "Nimbus empezó como un proyecto de fin de semana para compartir fotos entre tres amigos. "
        "Diez años después, repasamos los momentos que nos marcaron: el primer cliente de equipo, "
        "la llegada de la sincronización selectiva y la noche en que casi nos quedamos sin "
        "servidores.",
    ),
    (
        "es",
        "Guía de estilo para nombrar archivos",
        "Fechas al principio, sin espacios raros, versiones con número y no con la palabra "
        "«final». Reunimos las convenciones que más usan nuestros clientes para que los archivos "
        "se encuentren solos, incluso años después.",
    ),
    (
        "es",
        "Qué significa para nosotros la privacidad",
        "La privacidad no es una casilla en un formulario. En este texto explicamos los principios "
        "que guían nuestras decisiones de producto, desde el cifrado hasta el diseño de los "
        "permisos, sin entrar en los detalles legales de cada política.",
    ),
    (
        "es",
        "El año de la sincronización selectiva",
        "La sincronización selectiva fue la función más pedida durante dos años seguidos. Contamos "
        "por qué tardamos tanto en lanzarla, los prototipos que descartamos y qué nos enseñaron "
        "los primeros meses de uso real.",
    ),
    (
        "es",
        "Cinco hábitos para que la cuota de almacenamiento no te sorprenda",
        "Revisar la papelera de vez en cuando, desconfiar de las carpetas de descargas y archivar "
        "proyectos terminados: hábitos sencillos que mantienen ordenado cualquier espacio de "
        "almacenamiento, sea cual sea tu plan.",
    ),
    (
        "es",
        "Nimbus en las aulas",
        "Docentes de tres países nos contaron cómo usan carpetas compartidas para entregar "
        "trabajos, corregir y devolver comentarios. Una charla sobre educación, colaboración y la "
        "logística de cientos de archivos por semana.",
    ),
    (
        "es",
        "Cómo pensamos los planes",
        "Cada plan de Nimbus nace de una pregunta: ¿quién lo va a usar y para qué? Este artículo "
        "repasa la historia de nuestro catálogo, desde el primer plan gratuito hasta los "
        "complementos, sin entrar en precios ni límites concretos.",
    ),
    (
        "es",
        "Entrevista: la persona detrás del diseño de la papelera",
        "La papelera parece la parte más aburrida de un producto, hasta que alguien borra por error "
        "la tesis de un compañero. Conversamos con la diseñadora que rehízo la experiencia de "
        "eliminar y recuperar archivos.",
    ),
    (
        "es",
        "Nuestro compromiso con la accesibilidad",
        "Durante el último año auditamos todas las pantallas de Nimbus con lectores de pantalla y "
        "navegación por teclado. Compartimos lo que encontramos, lo que ya corregimos y lo que "
        "todavía está en camino.",
    ),
    (
        "es",
        "Cómo celebramos el día del soporte",
        "Una vez al año, todo el equipo, incluidas las personas de ingeniería, responde consultas "
        "de clientes durante un día. Este año alguien contestó su primera duda sobre facturas y "
        "otro aprendió qué es un contracargo.",
    ),
    (
        "es",
        "Trabajar con archivos grandes en conexiones lentas",
        "Muchos de nuestros clientes trabajan desde lugares con mala conexión. Contamos cómo "
        "probamos Nimbus en redes lentas simuladas y por qué la subida por fragmentos cambió la "
        "experiencia de quienes trabajan con video.",
    ),
    (
        "es",
        "El largo camino hacia la residencia de datos",
        "Ofrecer residencia de datos en varias regiones exigió rediseñar cómo guardamos cada "
        "archivo. Un recorrido técnico por las decisiones que tomamos y las que todavía "
        "discutimos.",
    ),
    (
        "es",
        "Lecciones de una caída",
        "El año pasado tuvimos una interrupción de servicio que afectó a la sincronización durante "
        "dos horas. Publicamos este análisis para contar qué pasó, cómo lo detectamos y qué "
        "cambiamos en nuestros procesos.",
    ),
    (
        "es",
        "Ideas para organizar el archivo familiar",
        "Fotos de cumpleaños, recetas de la abuela, papeles del auto. Algunas ideas para que el "
        "espacio compartido de la familia no se convierta en un cajón de sastre, sin importar qué "
        "plan tengan.",
    ),
    (
        "es",
        "Seguridad para equipos pequeños",
        "No hace falta un departamento de seguridad para proteger los archivos de una empresa "
        "pequeña. Repasamos prácticas razonables, desde las contraseñas hasta el control de "
        "dispositivos, pensadas para quien administra Nimbus a tiempo parcial.",
    ),
    (
        "es",
        "Glosario de Nimbus",
        "Sincronización, versión, enlace compartido, retención, licencia, complemento. Un "
        "glosario breve para quienes empiezan, con la definición informal de cada término y "
        "dónde aparece en la aplicación.",
    ),
    (
        "es",
        "Nimbus y el teletrabajo",
        "Cuando el equipo se reparte entre zonas horarias, los archivos compartidos se vuelven la "
        "memoria de la empresa. Tres clientes nos contaron cómo cambió su forma de trabajar "
        "cuando dejaron de enviarse adjuntos por correo.",
    ),
    (
        "es",
        "Del correo electrónico a las carpetas compartidas",
        "Enviar adjuntos parece cómodo hasta que hay cuatro versiones del mismo documento. Una "
        "guía amable para convencer a tu equipo de dar el salto, con ejemplos reales de clientes.",
    ),
    (
        "en",
        "How we think about pricing",
        "Pricing is a product decision, not just a finance one. This post explains the principles "
        "behind our plans and add-ons, and why we prefer simple tiers over endless options. It "
        "does not list current prices.",
    ),
    (
        "en",
        "Inside our incident response drills",
        "Twice a year we simulate a security incident end to end, from detection to customer "
        "communication. Here is how the drills are run and what the last one taught us.",
    ),
    (
        "en",
        "Customer story: a film studio on Nimbus Business",
        "A small film studio in Lisbon moved its editing workflow to Nimbus Business. They walked "
        "us through how they share dailies, handle huge uploads and keep clients in the loop.",
    ),
    (
        "en",
        "Designing the new sharing dialog",
        "Sharing a file should take one click, yet the old dialog had nine options. We tell the "
        "story of how we simplified it, which options we hid and which ones we had to bring back.",
    ),
    (
        "en",
        "A short history of our API",
        "Our public API started as an internal tool for the mobile apps. This post traces how it "
        "grew, how we version it and why webhooks arrived later than everyone expected.",
    ),
    (
        "en",
        "Why we invest in end-to-end encryption",
        "End-to-end encryption makes some features harder to build. We explain why we think it is "
        "worth it, and what it means for search, previews and account recovery.",
    ),
]

for i, (lang, title, body) in enumerate(BLOG, 1):
    add(f"blog-{i:02d}", f"{title}\n\n{body}", "blog", "noise", lang)


def write(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for d in DOCS:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("corpus.jsonl")
    write(out)
    print(f"wrote {len(DOCS)} documents to {out}")
