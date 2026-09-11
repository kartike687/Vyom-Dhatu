/**
 * Browser-ready Supabase Client for Static Command Centre UI
 * Configured with MOIL Manganese Digital Twin Project Credentials
 */

const SUPABASE_CONFIG = {
  url: "https://dkmxjeuoidbxnpwrjadv.supabase.co",
  anonKey: "sb_publishable_Y_MAxfihttak5yXTLta95w_0QgL7"
};

// Initialize if window.supabase is available via CDN or script tag
let supabaseClient = null;
if (typeof window !== 'undefined' && window.supabase && typeof window.supabase.createClient === 'function') {
  supabaseClient = window.supabase.createClient(SUPABASE_CONFIG.url, SUPABASE_CONFIG.anonKey);
}

window.SUPABASE_CONFIG = SUPABASE_CONFIG;
window.supabaseClient = supabaseClient;
